function NET_graph = generate_NET_graph(Madj)

    if any(Madj,'all')
        %figure
        first_case = 0;
        for i=1:1:size(Madj,1)
            for j=1:1:size(Madj,2)
                if i == 1 && Madj(i,j)==1 && first_case == 0
                    NET_graph = graph(i,j);
                    first_case = first_case + 1;
                elseif j>i && Madj(i,j)==1
                    NET_graph = addedge(NET_graph,i,j);
                end
            end
        end
    else
        NET_graph = [];
    end
end