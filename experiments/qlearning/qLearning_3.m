
%% 1. Localiza todos los metrics.csv partiendo de la carpeta “results” al lado de tu script
scriptDir  = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'experiment-3');      
files      = dir(fullfile(resultsDir, '**', 'metrics.csv'));

%% 2. Lee y concatena todas las tablas
All = table();
for k = 1:numel(files)
    % Lee el CSV (cada uno contiene 20 filas = 20 runs de esa configuración)
    T = readtable(fullfile(files(k).folder, files(k).name));
    
    % Añade al DataFrame maestro
    All = [All; T];  %#ok<AGROW>
end

%% 3.a Agrupa por (learning_rate, learning_rate_decay) y calcula el success_rate medio, recompensa media
[G, learning_rateVals, learning_rate_decayVals] = findgroups(All.learning_rate, All.learning_rate_decay);
succMean = splitapply(@mean, All.success_rate, G);
rewMean  = splitapply(@mean, All.mean_reward,  G);
stepsMean = splitapply(@mean, All.mean_steps,   G);
timeMean = splitapply(@mean, All.training_time,   G);

% Construye tabla resumen
Summary = table(learning_rateVals, learning_rate_decayVals, succMean, rewMean, stepsMean, timeMean, ...
                'VariableNames', {'learning_rate','learning_rate_decay','succMean', 'rewMean', 'stepsMean', 'timeMean'});

%% 3.b Calcula e imprime IC 95% para cada combinación
n     = 10;              % número de runs por combinación
alpha = 0.05;            % nivel de significación para 95% IC
tVal  = tinv(1 - alpha/2, n-1);  % t_{0.975,9}

fprintf('  alpha     alpha decay     Métrica       Media      IC_lower    IC_upper\n');
fprintf('----------------------------------------------------------------------\n');

for i = 1:height(Summary)
    a  = Summary.learning_rate(i);
    ad = Summary.learning_rate_decay(i);
    % máscara para esta combinación
    mask = All.learning_rate==a & All.learning_rate_decay==ad;

    % Success rate
    x = All.success_rate(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %.2f   Success-rate  %7.3f   [%6.3f, %6.3f]\n', ...
            a, ad, mu, mu-h, mu+h);

    % Recompensa media
    x = All.mean_reward(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %.2f   Rew. media    %7.3f   [%6.3f, %6.3f]\n', ...
            a, ad, mu, mu-h, mu+h);

    % Pasos medios
    x = All.mean_steps(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %.2f   Steps medios  %7.1f   [%6.1f, %6.1f]\n', ...
            a, ad, mu, mu-h, mu+h);

    % Tiempo medio
    x = All.training_time(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %.2f   Time (s)      %7.2f   [%6.2f, %6.2f]\n\n', ...
            a, ad, mu, mu-h, mu+h);
end

%% 4. Pivota para la heat-map
Learning_rateU = unique(Summary.learning_rate);                   % ejes X
Learning_rate_decayU  = unique(Summary.learning_rate_decay, 'stable'); % ejes Y
SuccMat = nan(numel(Learning_rate_decayU), numel(Learning_rateU));
RewMat  = nan(size(SuccMat));
StepMat  = nan(size(SuccMat));
TimeMat  = nan(size(SuccMat));

for i = 1:height(Summary)
    xi = find(Learning_rateU == Summary.learning_rate(i));
    yi = find(Learning_rate_decayU  == Summary.learning_rate_decay(i));
    SuccMat(yi, xi) = Summary.succMean(i);
    RewMat(yi,xi) = Summary.rewMean(i);
    StepMat(yi, xi) = Summary.stepsMean(i);
    TimeMat(yi, xi) = Summary.timeMean(i);
end

%% 5. Dibuja el heat-map de success-rate, recompensa media
figure
h = heatmap( ...
    Learning_rateU, ...                % etiquetas X = learning_rate
    Learning_rate_decayU, ...                 % etiquetas Y = alpha
    SuccMat, ...               % matriz de valores
    'Colormap', parula, ...
    'ColorLimits', [0 1] ...   % success_rate ∈ [0,1]
);
xlabel('\alpha')
ylabel('\alpha decay')
title('Success-rate medio por combinación')

figure
h1 = heatmap(Learning_rateU, Learning_rate_decayU, RewMat, ...
    'Colormap', parula, ...
    'ColorLimits', [min(RewMat(:)) max(RewMat(:))]);
xlabel('\alpha')
ylabel('\alpha decay')
title('Recompensa media por combinación')

figure
h1 = heatmap(Learning_rateU, Learning_rate_decayU, StepMat, ...
    'Colormap', parula, ...
    'ColorLimits', [min(StepMat(:)) max(StepMat(:))]);
xlabel('\alpha')
ylabel('\alpha decay')
title('Número de pasos medio por combinación')

%% 6. Boxplots de tiempo medio para las 3 mejores configuraciones
% Configuraciones hardcodeadas: [learning_rate, alpha]
best = [0.5, 0.99;
        0.5, 0.95;
        0.1,  0.995];

% Prepara vectores para el boxplot
times  = [];
groups = [];
labels = cell(size(best,1),1);

for i = 1:size(best,1)
    g = best(i,1);
    a = best(i,2);
    % Filtra en 'All' los runs de la config actual
    mask = All.learning_rate==g & All.learning_rate_decay==a;
    t   = All.training_time(mask);
    % Acumula
    times  = [times;  t];
    groups = [groups; repmat(i, numel(t), 1)];
    % Etiqueta "γ=… / a=…"
    labels{i} = sprintf('γ=%.2f, a=%.2f', g, a);
end

% Dibuja el boxplot
figure
boxplot(times, groups, ...
        'Labels', labels, ...
        'LabelOrientation','inline', ...
        'Whisker',1.5)   % puedes ajustar whisker si quieres
ylabel('Tiempo de entrenamiento (s)')
title('Distribución de tiempos (3 mejores config.)')
grid on