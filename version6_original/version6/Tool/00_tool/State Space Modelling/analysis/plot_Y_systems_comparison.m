function plot_Y_systems_comparison(abs_and_angle_1, abs_and_angle_2,max_freq, label1, label2)
    control = 1;

    karray1 = abs_and_angle_1.karray;
    abspp1 = abs_and_angle_1.abspp;
    abspn1 = abs_and_angle_1.abspn;
    absnp1 = abs_and_angle_1.absnp;
    absnn1 = abs_and_angle_1.absnn;

    anglepp1 = abs_and_angle_1.anglepp;
    anglepn1 = abs_and_angle_1.anglepn;
    anglenp1 = abs_and_angle_1.anglenp;
    anglenn1 = abs_and_angle_1.anglenn;

    karray2 = abs_and_angle_2.karray;
    abspp2 = abs_and_angle_2.abspp;
    abspn2 = abs_and_angle_2.abspn;
    absnp2 = abs_and_angle_2.absnp;
    absnn2 = abs_and_angle_2.absnn;

    anglepp2 = abs_and_angle_2.anglepp;
    anglepn2 = abs_and_angle_2.anglepn;
    anglenp2 = abs_and_angle_2.anglenp;
    anglenn2 = abs_and_angle_2.anglenn;

     % Create the tiled layout
    % figure('visible', 'on', 'Units', 'centimeters', 'Position', [4, 4, 20, 20]);
    figure('visible', 'on', 'Units', 'inches', 'Position', [1, 1, 3.5, 5]);

    t = tiledlayout(4, 2, "TileSpacing", "compact");

    % Plot each tile
    % Tile 1
    nexttile
    h1 = plot(karray1/(2*pi), abspp1(:, control), "LineWidth", 2, 'Color', 'k');
    hold on
    h2 = plot(karray2/(2*pi), abspp2(:, control), 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 1, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$|Y_{pp}|$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])

    % Tile 2
    nexttile
    plot(karray1/(2*pi), abspn1(:, control), "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), abspn2(:, control), 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 1, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$|Y_{pn}|$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Tile 3
    nexttile
    plot(karray1/(2*pi), anglepp1(:, control) * 180/pi, "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), anglepp2(:, control) * 180/pi, 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * -90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$\theta_{Y_{pp}}$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Tile 4
    nexttile
    plot(karray1/(2*pi), anglepn1(:, control) * 180/pi, "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), anglepn2(:, control) * 180/pi, 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * -90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$\theta_{Y_{pn}}$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Tile 5
    nexttile
    plot(karray1/(2*pi), absnp1(:, control), "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), absnp2(:, control), 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 1, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$|Y_{np}|$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Tile 6
    nexttile
    plot(karray1/(2*pi), absnn1(:, control), "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), absnn2(:, control), 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 1, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$|Y_{nn}|$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Tile 7
    nexttile
    plot(karray1/(2*pi), anglenp1(:, control) * 180/pi, "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), anglenp2(:, control) * 180/pi, 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * -90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$\theta_{Y_{np}}$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Tile 8
    nexttile
    plot(karray1/(2*pi), anglenn1(:, control) * 180/pi, "LineWidth", 2, 'Color', 'k');
    hold on
    plot(karray2/(2*pi), anglenn2(:, control) * 180/pi, 'LineStyle', '--', "LineWidth", 2, 'Color', [0.5, 0.5, 0.5]);
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * 90, 'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    plot(karray1/(2*pi), ones(1, size(karray1, 2)) * -90,  'LineStyle', '--', 'Color', [0.8500, 0.3250, 0.0980])
    hold off
    ylabel('$\theta_{Y_{nn}}$', 'Interpreter', 'Latex', 'FontSize', 11)
    xlabel('f [Hz]', 'Interpreter', 'Latex', 'FontSize', 11)
    xlim([0,max_freq])
    
    % Create the legend on the first axes and place it outside the plot area
    lgd = legend([h1, h2], {label1,label2}, ...
                 'Interpreter', 'Latex', 'FontSize', 11, 'Location', 'northoutside', ...
                 'Orientation', 'horizontal', 'NumColumns', 1);
    lgd.Layout.Tile = 'north';

    % Apply layout settings
    t.Padding = 'compact';
end
