classdef table
  properties
    VariableNames
    Data
  end
  methods
    function obj = table(data, varnames)
      if nargin > 0
        obj.Data = data;
        obj.VariableNames = varnames;
      end
    end
    function display(obj)
      disp('Table with properties:');
      disp(['    VariableNames: {' strjoin(obj.VariableNames, ', ') '}']);
      disp('    Data:');
      disp(obj.Data);
    end
  end
end

