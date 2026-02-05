# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:18:56 2025

@author: colives
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import copy
import GridCalEngine.api as gce
from GridCalEngine.DataStructures.numerical_circuit import compile_numerical_circuit_at
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import multi_island_pf_nc
from GridCalEngine.enumerations import ReactivePowerControlMode
from scipy.linalg import block_diag
from .utils_regression import *
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class ssscofo:
    
    def __init__(self, grid, ofo_W):
        
        self._grid = copy.deepcopy(grid)
        self._nb, self._nl = self._grid.get_dimensions()[0:-1]
        self._alpha = ofo_W['alpha']
        self._beta = ofo_W['beta']
        self._gamma = ofo_W['gamma']
        self._tol = ofo_W['tol']
        self._u_dict = ofo_W['u']
        self._y_dict = ofo_W['y']
        self._nu = self.count_u()
        self._ny = self.count_y()
        self._u = self.get_u(self._grid)
        self._y = self.meas_y(self._grid)
        self._Hu, self._Hy = self.build_H()
        self._A = self.build_A()
#        self._b = self.build_b()
        self._b = ofo_W['b']
        self._C = self.build_C()
#        self._d = self.build_d()
        self._d = ofo_W['d']
        self._theta = ofo_W['theta']
        self._f = self.f_and_penalty()
        self._G = ofo_W['G']
        self._ssconstraint = ofo_W['add constraint']
    
    def get_u(self, grid):
        aux = []
        for key in self._u_dict.keys():
            for k in self._u_dict[key]:
                if key == 'p':
                    aux.append(grid.generators[k].P/grid.generators[k].Snom)
                elif key == 'v':
                    aux.append(grid.generators[k].Vset)
        return np.array(aux)
    
    
    def count_u(self):
        aux = 0
        for key in self._u_dict.keys():
            aux += len(self._u_dict[key])
        return aux
    
    
    def meas_y(self, grid):
        pf = self.run_pf(grid)
        P = [pf.Sbus[k].real/grid.generators[k].Snom for k in self._y_dict['p']]
        Vm = [np.abs(pf.voltage[k]) for k in self._y_dict['v_m']]
        Va = [np.angle(pf.voltage[k]) for k in self._y_dict['v_a']]
        # print(f'{[pf.Sbus[k].real/grid.generators[k].Snom for k in [0,1,2]]}')
        return np.array(P+Vm+Va)
    
    
    def count_y(self):
        aux = 0
        for key in self._y_dict.keys():
            aux += len(self._y_dict[key])
        return aux
    
    
    def run_pf(self, grid):
        nc = compile_numerical_circuit_at(grid)
        pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, control_q=ReactivePowerControlMode.NoControl)#, max_iter=100)
        return multi_island_pf_nc(nc=nc, options=pf_options)
    
    
    def build_H(self):
        H_u = np.eye(self._nu)
        H_y = np.zeros((self._ny, self._nu))
        y_0 = np.copy(self.meas_y(self._grid))
        i = 0
        du = 0.01
        for key in self._u_dict.keys():
            for k in self._u_dict[key]:
                if key == 'p':
                    self._grid.generators[k].P += du*self._grid.generators[k].Snom
                    y = np.copy(self.meas_y(self._grid))
                    self._grid.generators[k].P -= du*self._grid.generators[k].Snom
                elif key == 'v':
                    self._grid.generators[k].Vset += du
                    y = np.copy(self.meas_y(self._grid))
                    self._grid.generators[k].Vset -= du
                hy = (y - y_0)/du
                H_y[:,i] = np.copy(hy)
                i+=1
        return H_u,H_y
    
    
    def build_A(self):
        aux = np.array([[1],[-1]])
        aux = np.kron(np.eye(self._nb + 2), aux)
        #ext = np.zeros(2*(self._nb + 2)).reshape((-1,1))
        return aux #np.concatenate((aux,ext,ext),axis=1)
        
        
#     def build_b(self):
#  #       return np.array([0.95, -0.2]+[0.95, -0.4] + [1.1, -0.9]*self._nb)
# #        return np.array([0.95, -0.2]+[0.95, -0.2] + [1.1, -0.9]*self._nb)
#         return np.array([0.95, -0.2]+[0.95, -0.2] + [1, -0.9]+[1.1, -0.9]*(self._nb-1))
    
    
    def build_C(self):
        aux = np.array([[1],[-1]])
        return np.kron(np.eye(self._nu), aux)
    
    
#     def build_d(self):
#         d = []
# #        d.extend([0.95, -0.4]+[0.95, -0.2])
#         d.extend([0.95, -0.2]+[0.95, -0.2])
#         for key in self._u_dict.keys():
#             # if key == 'p':
#             #     for k in self._u_dict[key]:
#             #         d.extend([0.95, -0.2])
#             #el
#             if key == 'v':
#                 d.extend([1.1, -0.9]*len(self._u_dict[key]))
#         d = np.array(d)
#         d[4]=1
#         return d
    
    
    # Regression: g(y) = -0.0155145364124157*max(0, 0.280713084794927 - theta1) - 0.0315442546471507*max(0, 1.08609116512098 - V1) - 0.000544874376064074*max(0, 1.21518076182698 - P_SG2) + 7.62429686999868e-5*max(0, 4.16138295634869 - P_GFOR1) + 0.00515434969172181*max(0, P_GFOR1 - 4.16138295634869) - 0.0002663633801353*max(0, P_SG2 - 1.21518076182698) - 0.0377954179176665*max(0, V1 - 1.08609116512098) + 0.00246455502475496*max(0, theta1 - 0.280713084794927) + 1.00696204466922
    def grad(self):
        Fu = np.zeros(self._nu)
        Fu[0] = 10*2*self._u[0]
        Fu[1] = 2*self._u[1]
        byp=np.zeros(self._ny)
        ayp=np.zeros(self._ny)
        byn=np.zeros(self._ny)
        ayn=np.zeros(self._ny)
        cy = np.zeros(self._ny)
        cz = np.zeros(3)
        z = np.zeros(3)
        bzp = np.copy(z)
        azp = np.copy(z)
        bzn = np.copy(z)
        azn = np.copy(z)
        y_r = np.copy(self._y)
        y_r[0] *= self._grid.generators[0].Snom/self._grid.Sbase
        y_r[1] *= self._grid.generators[1].Snom/self._grid.Sbase
        byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k= rgr_inputs(byp, ayp, byn, ayn, cy, cz, bzp,azp, bzn, azn, self._theta)
        
        Fy = small_signal_constraint_gradient(y_r, z, byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k)
        extra = np.zeros(self._ny)
        extra[0] = 2.0*self._y[0]
        return self._gamma*(Fu + self._Hy.T @ extra + self._beta*self._Hy.T @ Fy.T)
    
    def jac_ss_constr(self):
        byp=np.zeros(self._ny)
        ayp=np.zeros(self._ny)
        byn=np.zeros(self._ny)
        ayn=np.zeros(self._ny)
        cy = np.zeros(self._ny)
        cz = np.zeros(3)
        z = np.zeros(3)
        bzp = np.copy(z)
        azp = np.copy(z)
        bzn = np.copy(z)
        azn = np.copy(z)
        byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k= rgr_inputs(byp, ayp, byn, ayn, cy, cz, bzp,azp, bzn, azn, self._theta)

        y_r = np.copy(self._y)
        y_r[0] *= self._grid.generators[0].Snom/self._grid.Sbase
        y_r[1] *= self._grid.generators[1].Snom/self._grid.Sbase
        
        Fu = np.zeros(self._nu)
        
        t_0 = (y_r[2] - ayp[2])
        t_1 = (ayn[2] - y_r[2])

        Fu[2]= ((byp[2] * np.maximum(np.sign(t_0), 0)) - (byn[2] * np.maximum(np.sign(t_1), 0)))

        return Fu
    
    
    def opt_sgm(self):
        if self._G.size == 0:
            f = lambda z: (z + self.grad()) @ (z + self.grad())
            if self._ssconstraint:
                cons = ({'type':'ineq', 'fun': lambda z: self._d - self._C @ (self._u + self._alpha * z)},
                        {'type':'ineq', 'fun': lambda z: self._b - self._A @ (self._y + self._alpha * self._Hy @ z)},
                        {'type':'ineq', 'fun': lambda z: - self.g(self._y + self._alpha * self._Hy @ z)[1]})
            else:
                cons = ({'type':'ineq', 'fun': lambda z: self._d - self._C @ (self._u + self._alpha * z)},
                        {'type':'ineq', 'fun': lambda z: self._b - self._A @ (self._y + self._alpha * self._Hy @ z)})

            options={'maxiter':200}    
            res = minimize(f, np.zeros_like(self._u), jac = lambda z: 2*(z + self.grad()), method='SLSQP', constraints=cons, tol=self._tol, options=options)
            if not res.success:
                print(f'ofo problem: {res.message}')
        else:
            g = self.grad()      # ∇Phi(u,y), shape (p,)
            G = self._G         # SPD, shape (p,p)
            #H = self._Hy         # H(u), shape (?, p) so that H.T @ g is (p,)
         
            # Cholesky of G for stable solves and for the quadratic form
            cF = cho_factor(G)                           # G = L L^T
           # a  = H.T @ g                                 # a = H^T g
            c  = cho_solve(cF, g)                        # c = G^{-1} a
         
            def f(w):
                v = w + c
                # (w+c)^T G (w+c) computed stably
                # using Cholesky: ||L^T v||^2
                L = cF[0]
                lower = cF[1]
                y = (L.T @ v) if lower else (L @ v)
                return y @ y
         
            def f_grad(w):
                v = w + c
                # ∇ = 2 G (w+c)
                return 2.0 * (G @ v)
         
            if self._ssconstraint:
                cons = ({'type':'ineq', 'fun': lambda z: self._d - self._C @ (self._u + self._alpha * z)},
                        {'type':'ineq', 'fun': lambda z: self._b - self._A @ (self._y + self._alpha * self._Hy @ z)},
                        {'type':'ineq', 'fun': lambda z: - self.g(self._y + self._alpha * self._Hy @ z)[1]})#, 'jac': lambda z: self.jac_ss_constr()})
                        #{'type': 'ineq', 'fun': lambda z: - self.g(self._modified_input(z))[1]})
            else:
                cons = ({'type':'ineq', 'fun': lambda z: self._d - self._C @ (self._u + self._alpha * z)},
                        {'type':'ineq', 'fun': lambda z: self._b - self._A @ (self._y + self._alpha * self._Hy @ z)})


            options={'maxiter':200}
            res = minimize(f, np.zeros_like(self._u), method='SLSQP',
                           jac=f_grad, constraints=cons, tol=self._tol, options=options)
            if not res.success:
                print(f'ofo problem: {res.message}')
        #print(f'g(u,y)=',self.g(self._modified_input(res.x))[0])
        #print(f'g(y)=',self.g(self._y + self._alpha * self._Hy @ res.x)[0])
        return res.x
    
    # def _modified_input(self, z):
    #     x = np.copy(self._y + self._alpha * self._Hy @ z)
    #     x[2] = (self._u + self._alpha * z)[2]
    #     return x
    
    
    def updt_u(self, grid):
        self._y = self.meas_y(grid)
        du = self._alpha * self.opt_sgm()
        self._u += du
        return np.copy(self._u), np.max(np.abs(du))
    
    
    # def g(self):
    #     P_GFOR1 = self._y[0]*self._grid.generators[0].Snom/self._grid.Sbase
    #     P_SG2 = self._y[1]*self._grid.generators[1].Snom/self._grid.Sbase
    #     V1 = self._y[2]
    #     theta1 = self._y[11]
    #     return -0.0155145364124157*max(0, 0.280713084794927 - theta1) - 0.0315442546471507*max(0, 1.08609116512098 - V1) - 0.000544874376064074*max(0, 1.21518076182698 - P_SG2) + 7.62429686999868e-5*max(0, 4.16138295634869 - P_GFOR1) + 0.00515434969172181*max(0, P_GFOR1 - 4.16138295634869) - 0.0002663633801353*max(0, P_SG2 - 1.21518076182698) - 0.0377954179176665*max(0, V1 - 1.08609116512098) + 0.00246455502475496*max(0, theta1 - 0.280713084794927) + 1.00696204466922 
        
    def g(self, y):
        byp=np.zeros(self._ny)
        ayp=np.zeros(self._ny)
        byn=np.zeros(self._ny)
        ayn=np.zeros(self._ny)
        cy = np.zeros(self._ny)
        cz = np.zeros(3)
        z = np.zeros(3)
        bzp = np.copy(z)
        azp = np.copy(z)
        bzn = np.copy(z)
        azn = np.copy(z)
        # y_r = np.copy(self._y)
        # y_r[0] *= self._grid.generators[0].Snom/self._grid.Sbase
        # y_r[1] *= self._grid.generators[1].Snom/self._grid.Sbase

        y_r = np.copy(y)
        y_r[0] *= self._grid.generators[0].Snom/self._grid.Sbase
        y_r[1] *= self._grid.generators[1].Snom/self._grid.Sbase
        
        byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k= rgr_inputs(byp, ayp, byn, ayn, cy, cz, bzp,azp, bzn, azn, 0)
        
        return DI_rgr(y_r, z, byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k, self._theta)
        
    
    def f_and_penalty(self):
        f = self._y[0]**2+10*self._u[0]**2+self._u[1]**2
        DI, penalty = self.g(self._y)
        return f, penalty, f+penalty*self._beta
    
class ssscopf:
    
    def __init__(self, grid, params):
        
        self._grid = copy.deepcopy(grid)
        self._nb, self._nl = self._grid.get_dimensions()[0:-1]
        self._beta = params['beta']
        self._gamma = params['gamma']
        self._tol = params['tol']
        self._u_dict = params['u']
        self._y_dict = params['y']
        self._ny = self.count_y()
        self._nu = self.count_u()
        self._A = self.build_A()
        #self._b = self.build_b()
        self._b = params['b']
        self._x = self.x_init()
        self._theta = params['theta']
        self._ssconstraint = params['add constraint']
        self._slack_gen_position = params['slack_gen_position']
        self._generator_cost_2 = params['generator_cost_2']
        
    def count_y(self):
        aux = 0
        for key in self._y_dict.keys():
            aux += len(self._y_dict[key])
        return aux
    
    
    def count_u(self):
        aux = 0
        for key in self._u_dict.keys():
            aux += len(self._u_dict[key])
        return aux

    
    def build_A(self):
        aux = np.array([[1],[-1]])
        aux = np.kron(np.eye(len(self._y_dict['p'])+len(self._y_dict['v_m'])), aux)
        if len(self._y_dict['v_a'])>0:
            ext = np.zeros([2*(len(self._y_dict['p'])+len(self._y_dict['v_m'])),len(self._y_dict['v_a'])])#.reshape((-1,1))
            Ay = np.concatenate((aux,ext),axis=1)

        else:
            Ay = aux

        aux = np.array([[1],[-1]])
        Au = np.kron(np.eye(self._nu), aux)
        return  block_diag(Ay,Au)
        
        
#     def build_b(self):
#         #by = np.array([0.95, -0.2]+[0.95, -0.4] + [1.1, -0.9]*self._nb)
#         by = np.array([0.95, -0.2]+[0.95, -0.2] + [1.1, -0.9] + [1.1, -0.9]*(self._nb-1))
#         bu = []
# #        bu.extend([0.95, -0.4]+[0.95,-0.2])
#         bu.extend([0.95, -0.2]+[0.95,-0.2])
#         for key in self._u_dict.keys():
#             # if key == 'p':
#             #     for k in self._u_dict[key]:
#             #         bu.extend([0.95, -0.2])
            
#             #el
#             if key == 'v':
#                 bu.extend([1.1, -0.9]*len(self._u_dict[key]))
#         bu = np.array(bu)
#         bu[4] =1.1
#         return np.concatenate((by,bu))
        
    
    def x_init(self):
        nc = compile_numerical_circuit_at(self._grid)
        pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, control_q=ReactivePowerControlMode.NoControl)#, max_iter=100)
        pf = multi_island_pf_nc(nc=nc, options=pf_options)
        P=[]
        for i,k in enumerate(self._y_dict['p']):
            try:
                Pl=self._grid.buses[k].loads.P
            except:
                Pl=0
            P.append((pf.Sbus[k].real+Pl)/self._grid.generators[i].Snom)
        Vm = [np.abs(pf.voltage[k]) for k in self._y_dict['v_m']]
        Va = [np.angle(pf.voltage[k]) for k in self._y_dict['v_a']]
        Pg = [self._grid.generators[k].P/self._grid.generators[k].Snom for k in self._u_dict['p']]
        Vg = [self._grid.generators[k].Vset for k in self._u_dict['v']]
        return np.array(P+Vm+Va+Pg+Vg)
        
        
    
    def h(self, u):
        for idx, u_p in enumerate(u[:len(self._u_dict['p'])]):
            if idx < self._slack_gen_position:
                self._grid.generators[idx].P = u_p*self._grid.generators[idx].Snom
            else:
                self._grid.generators[idx+1].P = u_p*self._grid.generators[idx+1].Snom

        for idx, u_v in enumerate(u[len(self._u_dict['p']):]):
            self._grid.generators[idx].Vset = u_v
        nc = compile_numerical_circuit_at(self._grid)
        pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, control_q=ReactivePowerControlMode.NoControl)#, max_iter=100)
        pf = multi_island_pf_nc(nc=nc, options=pf_options)
        P=[]
        for i,k in enumerate(self._y_dict['p']):
            try:
                Pl=self._grid.buses[k].loads.P
            except:
                Pl=0
            P.append((pf.Sbus[k].real+Pl)/self._grid.generators[i].Snom)        
        Vm = [np.abs(pf.voltage[k]) for k in self._y_dict['v_m']]
        Va = [np.angle(pf.voltage[k]) for k in self._y_dict['v_a']]
        return np.array(P+Vm+Va)
    
    
    def g(self, x):
        byp=np.zeros(self._ny)
        ayp=np.zeros(self._ny)
        byn=np.zeros(self._ny)
        ayn=np.zeros(self._ny)
        cy = np.zeros(self._ny)
        cz = np.zeros(3)
        
        y_r = np.copy(x[:self._ny])
        y_r[0] *= self._grid.generators[0].Snom/self._grid.Sbase
        y_r[1] *= self._grid.generators[1].Snom/self._grid.Sbase
        
        z = np.zeros(3)
        bzp = np.copy(z)
        azp = np.copy(z)
        bzn = np.copy(z)
        azn = np.copy(z)
        y_r = np.copy(x[:self._ny])
        byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k= rgr_inputs(byp, ayp, byn, ayn, cy, cz, bzp,azp, bzn, azn, 0)
        
        return DI_rgr(y_r, z, byp, ayp, byn, ayn, cy, cz, bzp, azp, bzn, azn, k, self._theta)


    def coupling_constraint(self, x, slack_gen_position):
        n = len(self._y_dict['p']) + 1
    
        idx_left = np.arange(n)
        idx_left = idx_left[idx_left != slack_gen_position]   # skip 10
    
        idx_right = np.where(idx_left < slack_gen_position,
                              self._ny + idx_left,
                              self._ny + idx_left - 1)
    
        return x[idx_left] - x[idx_right]
    


    def optimisation(self):
        mtd = 'SLSQP' # 'trust-constr' #
#        f = lambda x: (10*x[12]**2 + x[13]**2 + x[0]**2 + self._beta * self.g(x)[1])*self._gamma
        if self._beta !=0:
            f = lambda x: (sum(self._generator_cost_2*x[:len(self._y_dict['p'])]**2) + self._beta * self.g(x)[1])*self._gamma
        else:
            f = lambda x: (sum(self._generator_cost_2*x[:len(self._y_dict['p'])]**2))*self._gamma

        if self._ssconstraint == True:
            if mtd == 'SLSQP':
                cons = ({'type':'ineq', 'fun': lambda x: self._b - self._A @ x},
                        {'type':'eq', 'fun': lambda x: x[:self._ny] - self.h(x[self._ny:])},
                        {'type':'eq', 'fun': lambda x, slack_gen_position=self._slack_gen_position: self.coupling_constraint(x, slack_gen_position)},
                        {'type':'ineq', 'fun': lambda x: - self.g(self.h(x[self._ny:]))[1]})
            # elif mtd == 'trust-constr':
            #     lc1 = LinearConstraint(np.array([0,1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0]),0,0)
            #     lc2 = LinearConstraint(self._A[::2,:], -self._b[1::2], self._b[::2])
            #     cons2 = lambda x: x[:self._ny] - self.h(x[self._ny:])
            #     nlc2 = NonlinearConstraint(cons2,np.zeros(self._ny),np.zeros(self._ny))
            #     cons = [lc1, lc2, nlc2]
            options={'maxiter':200}    
        else:
            if mtd == 'SLSQP':
                cons = ({'type':'ineq', 'fun': lambda x: self._b - self._A @ x},
                        {'type':'eq', 'fun': lambda x: x[:self._ny] - self.h(x[self._ny:])},
                        {'type':'eq', 'fun': lambda x, slack_gen_position=self._slack_gen_position: self.coupling_constraint(x, slack_gen_position)})
            # elif mtd == 'trust-constr':
            #     lc1 = LinearConstraint(np.array([0,1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0]),0,0)
            #     lc2 = LinearConstraint(self._A[::2,:], -self._b[1::2], self._b[::2])
            #     cons2 = lambda x: x[:self._ny] - self.h(x[self._ny:])
            #     nlc2 = NonlinearConstraint(cons2,np.zeros(self._ny),np.zeros(self._ny))
            #     cons = [lc1, lc2, nlc2]
            options={'maxiter':200}    


        res = minimize(f, self._x, method=mtd, constraints=cons, tol=self._tol, options=options)
        print(f'opf: << Optimisation succed: {res.success} >>')
        if not res.success:
            print(res.message)
        self._x = res.x
        return res.x, res.success, res.nit
    
    

def execute_opf(grid, OPF_prms, print_opt=True):
    opf = ssscopf(grid, OPF_prms)

    res_opf, convergence, n_iter = opf.optimisation()
    opf_res = np.copy(res_opf)

    np.set_printoptions(precision=3)
    if OPF_prms['add constraint']==False:
        print('\n--- OPF Results ---')
        if OPF_prms['beta']!=0:
            print('\n--- SSC-OPF Penalty Results ---')
    else:
        print('\n--- SSC-OPF Results ---')
    
    obj_fun_val=  sum((opf_res[:len(OPF_prms['y']['p'])])**2*OPF_prms['generator_cost_2'])
    for i, k in enumerate(OPF_prms['y']['p']):
        opf_res[i] *= grid.generators[i].Snom
    for i, k in enumerate(OPF_prms['u']['p']):
        opf_res[i+ssscopf.count_y(opf)] *= grid.generators[k].Snom
    
    if print_opt == True:
       # print(f'y_opf = {opf_res[:'+str(ssscopf.count_y(opf))+']}')
       # print(f'u_opf = {opf_res['+str(ssscopf.count_y(opf))+':]}')
        print('obj fun = ', obj_fun_val)

    for idx, gen in enumerate(grid.generators):
        gen.P = opf_res[idx]
        gen.Vset = opf_res[idx+len(grid.generators)]

    nc = compile_numerical_circuit_at(grid)    
    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8, control_q=ReactivePowerControlMode.NoControl)#, max_iter=100)
    opf_sol_completa = multi_island_pf_nc(nc=nc, options=pf_options)

    # print('\n--- OPF Complete Solution---')
    # print(opf_sol_completa.Sbus)
    # print(np.abs(opf_sol_completa.voltage))
    
    return opf_res, opf_sol_completa, opf, convergence, n_iter
