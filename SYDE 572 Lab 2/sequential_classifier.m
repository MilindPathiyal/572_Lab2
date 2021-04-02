% Loads 2 Class of Data A and B
load('lab2_3.mat');
cls_key = {'a','b'};
cls_id = [1, 2];
cls_map = containers.Map(cls_key,cls_id);

% Define grid
resolution = 1;

min_x = min([min(a(:,1)) min(b(:,1))]) - resolution;
max_x = max([max(a(:,1)) max(b(:,1))]) + resolution;
min_y = min([min(a(:,2)) min(b(:,2))]) - resolution;
max_y = max([max(a(:,2)) max(b(:,2))]) + resolution;
x_range = min_x:resolution:max_x;
y_range = min_y:resolution:max_y;
[X,Y] = meshgrid(x_range, y_range);

%% Compute Sequential Discrimnants with MED Classifier
errors_list = {};
for n_iter=1:N
    subset_a = a;
    subset_b = b;
    discriminant_list = {};
    n_ab_list = {};
    n_ba_list = {};
    errors = [];
    j = 0;
    while( j < J && ~isempty(subset_a) && ~isempty(subset_b))
        valid_discriminate = false;
        [n_ab, n_ba] = deal(zeros(1), zeros(1));

        while(~isempty(n_ab) && ~isempty(n_ba))
            rd_row_a = randi(size(subset_a,1), 1);
            rd_row_b = randi(size(subset_b,1), 1);
            prot_a = subset_a(rd_row_a, :);
            prot_b = subset_b(rd_row_b, :);
            
            G = zeros(size(X));
            for i=1:numel(X)
                p = [X(i) Y(i)];

                d_to_a = (p(1) - prot_a(1))^2 + (p(2) - prot_a(2))^2;
                d_to_b = (p(1) - prot_b(1))^2 + (p(2) - prot_b(2))^2;

                [min_val, cls_index] = min([d_to_a d_to_b]);
                G(i) = cls_index;
            end
    
            [n_aa, n_ab] = evaluate_predictions(subset_a, G, X, Y, cls_map('a'));
            [n_bb, n_ba] = evaluate_predictions(subset_b, G, X, Y, cls_map('b'));
        end

        discriminant_list{end+1} = G;
        n_ab_list{end+1} = n_ab;
        n_ba_list{end+1} = n_ba;
        if plot_error
            err_rate = (length(n_ab)+length(n_ba)) / 400;
            errors = [errors err_rate];
        end

        if isempty(n_ab); subset_b = remove_correct(subset_b, n_bb); end
        if isempty(n_ba); subset_a = remove_correct(subset_a, n_aa); end
        j = j+1;
    end

    if plot_boundary        
        
        grid_SC = zeros(size(discriminant_list{1}));
        for i=1:numel(grid_SC)
            for i_d=1:length(discriminant_list)
                G = discriminant_list{i_d};
                n_ab = n_ab_list{i_d};
                n_ba = n_ba_list{i_d};

                if G(i) == cls_map('b') && isempty(n_ab)
                    grid_SC(i) = cls_map('b');
                    break
                end

                if G(i) == cls_map('a') && isempty(n_ba)
                    grid_SC(i) = cls_map('a');
                    break
                end

            end
        end
        classes = {a,b};       
        figure;
        hold on;
        
        map = [
            1, 0.5, 0.5
            0.5, 0.5, 1
            ];
        colormap(map); 
        contourf(X, Y, grid_SC);

        x = classes{1}(:,1);
        y = classes{1}(:,2);
        scatter(x,y,25,'r','+');
    
        x = classes{2}(:,1);
        y = classes{2}(:,2);
        scatter(x,y,25,'b','*');
        
        legend('Sequential Discriminants Classifier Decision Boundary', 'Class A', 'Class B')
        title('Sequential  Discriminants Classifier Decision Boundary');
        hold off;
        
    end
    
    if plot_error
        errors_list{end+1} = errors;
    end
end

if plot_error
    %plot_error_rates(errors_list);
    plot_types = {
        'a) Average Error Rate',...
        'b) Minimum Error Rate',...
        'c) Maximum Error Rate',... 
        'd) Standard Deviation of Error Rate'};
    [avg_err, min_err, max_err, std_err] = deal([],[],[],[]);
    
    N = length(errors_list);
    J = length(errors_list{1});
    for j=1:J
        errs = zeros(1, N);
        for i=1:length(errors_list)
            errs(i) = errors_list{i}(j);
        end
        
        avg_err = [avg_err mean(errs)];
        min_err = [min_err min(errs)];
        max_err = [max_err max(errs)];
        std_err = [std_err std(errs)];
    end
    processed_errs = {avg_err, min_err, max_err, std_err};
    
    figure;
    for i=1:length(plot_types)
        subplot(length(plot_types),1,i);
        plot(1:J, processed_errs{i}, 'o-','linewidth',2,'markersize',5,'markerfacecolor','r');
        title(plot_types(i));   
    end
end

function set_ = remove_correct(set_, correct_points_rows)
    % Starts at the last row to avoid row index mutation within set
    for i=length(correct_points_rows):-1:1
        set_(correct_points_rows(i),:) = [];
    end
end
