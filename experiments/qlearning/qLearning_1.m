
%% 1. Localiza todos los metrics.csv partiendo de la carpeta “results” al lado de tu script
scriptDir  = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'experiment-1');      
files      = dir(fullfile(resultsDir, '**', 'metrics.csv'));

%% 2. Lee y concatena todas las tablas
All = table();
for k = 1:numel(files)
    % Lee el CSV (cada uno contiene 20 filas = 20 runs de esa configuración)
    T = readtable(fullfile(files(k).folder, files(k).name));
    
    % Añade al DataFrame maestro
    All = [All; T];  %#ok<AGROW>
end

%% 3. Agrupa por (gamma, learning_rate) y calcula el success_rate medio, recompensa media
[G, gammaVals, learningRateVals] = findgroups(All.gamma, All.learning_rate);
succMean = splitapply(@mean, All.success_rate, G);
rewMean  = splitapply(@mean, All.mean_reward,  G);
stepsMean = splitapply(@mean, All.mean_steps,   G);
timeMean = splitapply(@mean, All.training_time,   G);

% Construye tabla resumen
Summary = table(gammaVals, learningRateVals, succMean, rewMean, stepsMean, timeMean, ...
                'VariableNames', {'gamma','learning_rate','succMean', 'rewMean', 'stepsMean', 'timeMean'});

%% 4. Pivota para la heat-map
GammaU = unique(Summary.gamma);                   % ejes X
LearningRateU  = unique(Summary.learning_rate, 'stable'); % ejes Y
SuccMat = nan(numel(LearningRateU), numel(GammaU));
RewMat  = nan(size(SuccMat));
StepMat  = nan(size(SuccMat));
TimeMat  = nan(size(SuccMat));

for i = 1:height(Summary)
    xi = find(GammaU == Summary.gamma(i));
    yi = find(LearningRateU  == Summary.learning_rate(i));
    SuccMat(yi, xi) = Summary.succMean(i);
    RewMat(yi,xi) = Summary.rewMean(i);
    StepMat(yi, xi) = Summary.stepsMean(i);
    TimeMat(yi, xi) = Summary.timeMean(i);
end

%% 5. Dibuja el heat-map de success-rate, recompensa media
figure
h = heatmap( ...
    GammaU, ...                % etiquetas X = gamma
    LearningRateU, ...                 % etiquetas Y = alpha
    SuccMat, ...               % matriz de valores
    'Colormap', parula, ...
    'ColorLimits', [0 1] ...   % success_rate ∈ [0,1]
);
xlabel('\gamma')
ylabel('\alpha')
title('Success-rate medio por combinación')

figure
h1 = heatmap(GammaU, LearningRateU, RewMat, ...
    'Colormap', parula, ...
    'ColorLimits', [min(RewMat(:)) max(RewMat(:))]);
xlabel('\gamma')
ylabel('# \alph')
title('Recompensa media por combinación')

figure
h1 = heatmap(GammaU, LearningRateU, StepMat, ...
    'Colormap', parula, ...
    'ColorLimits', [min(StepMat(:)) max(StepMat(:))]);
xlabel('\gamma')
ylabel('\alpha')
title('Número de pasos medio por combinación')

