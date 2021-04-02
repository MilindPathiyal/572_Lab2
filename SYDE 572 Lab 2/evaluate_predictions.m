function [correct_pred, false_pred] = evaluate_predictions(data, grid, X, Y, actual)
    correct_pred = [];
    false_pred = [];
    for row_index=1:size(data, 1)
        [row, col] = find_grid_pos(X, Y, data(row_index,:));
        predicition = grid(row, col);
        
        if predicition ~= actual
            false_pred = [false_pred row_index];
        else
            correct_pred = [correct_pred row_index];
        end
    end
end

function [row, col] = find_grid_pos(X, Y, point)
    x = point(1);
    y = point(2);
    x_min = X(1);
    y_min = Y(1);
    
    row = y - y_min + 1;
    col = x - x_min + 1;
end