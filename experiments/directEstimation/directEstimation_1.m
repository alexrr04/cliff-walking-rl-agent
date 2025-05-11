%% 1. Localiza todos los metrics.csv partiendo de la carpeta “results” al lado de tu script
scriptDir  = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'experiment-1');      
files      = dir(fullfile(resultsDir, '**', 'metrics.csv'));

%% 2. Lee y concatena todas las tablas
All = table();
for k = 1:numel(files)
    % Lee el CSV (cada uno contiene 10 filas = 10 runs de esa configuración)
    T = readtable(fullfile(files(k).folder, files(k).name));
    
    % Añade al DataFrame maestro
    All = [All; T];  %#ok<AGROW>
end

%% 3.a Agrupa por (gamma, num_trajectories) y calcula el success_rate medio, recompensa media
[G, gammaVals, trajVals] = findgroups(All.gamma, All.num_trajectories);
succMean = splitapply(@mean, All.success_rate, G);
rewMean  = splitapply(@mean, All.mean_reward,  G);
stepsMean = splitapply(@mean, All.mean_steps,   G);
timeMean = splitapply(@mean, All.training_time,   G);

% Construye tabla resumen
Summary = table(gammaVals, trajVals, succMean, rewMean, stepsMean, timeMean, ...
                'VariableNames', {'gamma','num_trajectories','succMean', 'rewMean', 'stepsMean', 'timeMean'});

%% 3.b Calcula e imprime IC 95% para cada combinación
n     = 10;              % número de runs por combinación
alpha = 0.05;            % nivel de significación para 95% IC
tVal  = tinv(1 - alpha/2, n-1);  % t_{0.975,9}

fprintf('  γ     traj     Métrica       Media      IC_lower    IC_upper\n');
fprintf('---------------------------------------------------------------\n');

for i = 1:height(Summary)
    g  = Summary.gamma(i);
    tr = Summary.num_trajectories(i);
    % máscara para esta combinación
    mask = All.gamma==g & All.num_trajectories==tr;
    
    % Success rate
    x = All.success_rate(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %5d   Success-rate  %7.3f   [%6.3f, %6.3f]\n', ...
            g, tr, mu, mu-h, mu+h);
    
    % Recompensa media
    x = All.mean_reward(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %5d   Rew. media    %7.3f   [%6.3f, %6.3f]\n', ...
            g, tr, mu, mu-h, mu+h);
    
    % Pasos medios
    x = All.mean_steps(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %5d   Steps medios  %7.1f   [%6.1f, %6.1f]\n', ...
            g, tr, mu, mu-h, mu+h);
    
    % Tiempo medio
    x = All.training_time(mask);
    mu = mean(x); s = std(x);
    h = tVal * s / sqrt(n);
    fprintf(' %.2f   %5d   Time (s)      %7.2f   [%6.2f, %6.2f]\n\n', ...
            g, tr, mu, mu-h, mu+h);
end

%% 4. Pivota para la heat-map
GammaU = unique(Summary.gamma);                   % ejes X
TrajU  = unique(Summary.num_trajectories, 'stable'); % ejes Y
SuccMat = nan(numel(TrajU), numel(GammaU));
RewMat  = nan(size(SuccMat));
StepMat  = nan(size(SuccMat));
TimeMat  = nan(size(SuccMat));

for i = 1:height(Summary)
    xi = find(GammaU == Summary.gamma(i));
    yi = find(TrajU  == Summary.num_trajectories(i));
    SuccMat(yi, xi) = Summary.succMean(i);
    RewMat(yi,xi) = Summary.rewMean(i);
    StepMat(yi, xi) = Summary.stepsMean(i);
    TimeMat(yi, xi) = Summary.timeMean(i);
end

%% 5. Dibuja el heat-map de success-rate, recompensa media
figure
h = heatmap( ...
    GammaU, ...                % etiquetas X = gamma
    TrajU, ...                 % etiquetas Y = num_trajectories
    SuccMat, ...               % matriz de valores
    'Colormap', parula, ...
    'ColorLimits', [0 1] ...   % success_rate ∈ [0,1]
);
xlabel('\gamma')
ylabel('# Trayectorias')
title('Success-rate medio por combinación')

figure
h1 = heatmap(GammaU, TrajU, RewMat, ...
    'Colormap', parula, ...
    'ColorLimits', [min(RewMat(:)) max(RewMat(:))]);
xlabel('\gamma')
ylabel('# Trayectorias')
title('Recompensa media por combinación')

figure
h1 = heatmap(GammaU, TrajU, StepMat, ...
    'Colormap', parula, ...
    'ColorLimits', [min(StepMat(:)) max(StepMat(:))]);
xlabel('\gamma')
ylabel('# Trayectorias')
title('Número de pasos medio por combinación')

%% 6. Boxplots de tiempo medio para las 3 mejores configuraciones
% Configuraciones hardcodeadas: [gamma, num_trajectories]
best = [0.99,  500;
        0.99, 1000;
        0.95, 1000];

% Prepara vectores para el boxplot
times  = [];
groups = [];
labels = cell(size(best,1),1);

for i = 1:size(best,1)
    g = best(i,1);
    t = best(i,2);
    % Filtra en 'All' los runs de la config actual
    mask = All.gamma==g & All.num_trajectories==t;
    tt   = All.training_time(mask);
    % Acumula
    times  = [times;  tt];
    groups = [groups; repmat(i, numel(tt), 1)];
    % Etiqueta "γ=… / n=…"
    labels{i} = sprintf('γ=%.2f, n=%d', g, t);
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