% Given coordinates for points 1 to 4
x1 = -10;
y1 = 0;
x2 = 0;
y2 = -10;
x3 = 10;
y3 = 0;
x4 = 0;
y4 = 10;

% Define initial state estimate and covariance
X0 = [-9.76476; -9.76476; 1; 1]; % Initial estimate of (x, y) position and (vx, vy) velocity
P = diag([1,100, 500, 1000]); % Initial covariance matrix

% Define process noise covariance
Q = diag([0.1, 0.1, 0.1, 1]); % Process noise covariance

% Define measurement noise covariance
R = diag([0.1, 0.1, 0.1, 0.1]); % Measurement noise covariance

% Initialize arrays to store trace of covariance matrix
trace_P_history = zeros(20, 1);
X_history = zeros(20, 1);
Y_history = zeros(20, 1);

% Set delta t
del_t = 1;

% Define Lb matrix
d1 = [8.77, 7.97, 7.34, 6.93, 6.79,6.93, 7.34, 7.97, 8.77, 9.72, 10.76, 11.88, 13.06, 14.28, 15.53, 16.81, 18.10, 19.42, 20.74, 22.08];
d2 = [8.77, 7.97, 7.34, 6.93, 6.79, 6.93, 7.34, 7.97, 8.77, 9.72, 10.76, 11.88, 13.06, 14.28, 15.53, 16.81, 18.10, 19.42, 20.74, 22.08];
d3 = [20.74, 19.42, 18.10, 16.81, 15.53, 14.28, 13.06, 11.88, 10.76, 9.72, 8.77, 7.97, 7.34, 6.93, 6.79, 6.93, 7.34, 7.97, 8.77, 9.72];
d4 = [20.74, 19.42, 18.10, 16.81, 15.53, 14.28, 13.06, 11.88, 10.76, 9.72, 8.77, 7.97, 7.34, 6.93, 6.79, 6.93, 7.34, 7.97, 8.77, 9.72];
Lb = [d1; d2; d3; d4];

% Loop over each Lb value
for i = 1:20
    % Set the current Lb value
    current_Lb = Lb(:, i);

    % State transition model
    fk_1 = [1, 0, del_t, 0;
            0, 1, 0, del_t;
            0, 0, 1, 0;
            0, 0, 0, 1];

    % Predict state estimate
    Xk_min = fk_1 * X0;

    % Measurement model
    d1 = sqrt((Xk_min(1) - x1)^2 + (Xk_min(2) - y1)^2);
    d2 = sqrt((Xk_min(1) - x2)^2 + (Xk_min(2) - y2)^2);
    d3 = sqrt((Xk_min(1) - x3)^2 + (Xk_min(2) - y3)^2);
    d4 = sqrt((Xk_min(1) - x4)^2 + (Xk_min(2) - y4)^2);
    F = [d1; d2; d3; d4]; % Array of the observation equations

    % Jacobian of measurement model
    syms x y Vx Vy;
    F_sym = [sqrt((x - x1)^2 + (y - y1)^2); sqrt((x - x2)^2 + (y - y2)^2); ...
             sqrt((x - x3)^2 + (y - y3)^2); sqrt((x - x4)^2 + (y - y4)^2)];
    Hk_sym = jacobian(F_sym, [x, y, Vx, Vy]);
    Hk = double(subs(Hk_sym, [x, y, Vx, Vy], [X0(1), X0(2), X0(3), X0(4)]));

    % Process noise covariance matrix
    Qk_1 = diag([0.1, 0.1, 0.1, 0.1]);

    % Measurement noise covariance matrix
    Rk = diag([1, 1, 1, 1]);

    % Kalman filter update
    Zk = current_Lb + (Hk * (X0 - Xk_min));
    Kk = P * Hk' * inv((Hk * P * Hk') + Rk);
    P = (eye(4) - (Kk * Hk)) * P * (eye(4) - (Kk * Hk))' + (Kk * Rk * Kk');
    X0 = Xk_min + (Kk * (current_Lb - Zk));

    % Store the trace of covariance matrix
    trace_P_history(i) = trace(P);
    
    % Store x and y values
    X_history(i) = X0(1);
    Y_history(i) = X0(2);
    
    % Display the results for this iteration
    fprintf('For Lb value %d:\n', i);
    fprintf('X%d = %.6f\n', i, X0(1));
    fprintf('Y%d = %.6f\n\n', i, X0(2));
end

%{
}% Plot covariance versus time
subplot(2, 1, 1);
plot(1:20, trace_P_history, '-o');
xlabel('Time');
ylabel('Trace of Covariance Matrix');
title('Trace of Covariance Matrix vs. Time');

% Plot x versus y
subplot(2, 1, 2);
plot(X_history, Y_history, '-o');
xlabel('X');
ylabel('Y');
title('X vs Y');
%}

subplot(2, 1, 1);
plot(1:20, trace_P_history, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'red');
xlabel('Time');
ylabel('Trace of Covariance Matrix');
title('Trace of Covariance Matrix vs. Time');
grid on;  % Add grid lines

subplot(2, 1, 2);
plot(X_history, Y_history, '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'blue');
xlabel('X');
ylabel('Y');
title('X vs Y');
grid on;  % Add grid lines

