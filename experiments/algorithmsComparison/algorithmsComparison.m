%% 0. Configuración de rutas y algoritmos
scriptDir = fileparts(mfilename('fullpath'));
baseDir   = fullfile(scriptDir, 'final_experiment');  
% Debe contener tres subcarpetas, p.ej.:
% final_experiment/value_iteration/metrics.csv
% final_experiment/direct_estimation/metrics.csv
% final_experiment/q_learning/metrics.csv
algos = {'value_iteration', 'direct_estimation', 'q_learning'};

%% 1. Leer y concatenar todos los metrics.csv
All = table();
for a = 1:numel(algos)
    algo = algos{a};
    csvPath = fullfile(baseDir, algo, 'metrics.csv');
    if ~isfile(csvPath)
        error('No existe %s', csvPath)
    end
    T = readtable(csvPath);
    T.Algorithm = repmat({algo}, height(T), 1);
    All = [All; T];  %#ok<AGROW>
end

%% 2. Tabla resumen: media ± std por algoritmo
metrics = {'success_rate','mean_reward','mean_steps','training_time'};
nAlgos  = numel(algos);

mu   = zeros(nAlgos, numel(metrics));
sigma= zeros(nAlgos, numel(metrics));

for i = 1:nAlgos
    mask = strcmp(All.Algorithm, algos{i});
    for j = 1:numel(metrics)
        v = All.(metrics{j})(mask);
        mu(i,j)    = mean(v);
        sigma(i,j) = std(v);
    end
end

% Crea la tabla
Summary = table(algos(:), ...
    mu(:,1), sigma(:,1), ...
    mu(:,2), sigma(:,2), ...
    mu(:,3), sigma(:,3), ...
    mu(:,4), sigma(:,4), ...
    'VariableNames', { ...
      'Algorithm', ...
      'SuccMean','SuccStd', ...
      'RewMean','RewStd', ...
      'StepsMean','StepsStd', ...
      'TimeMean','TimeStd'});

disp('Resumen métricas (media ± std) por algoritmo:')
disp(Summary)

%% 3. Boxplots comparativos
% Para cada métrica generamos una figura
for j = 1:numel(metrics)
    figure('Name', metrics{j}, 'NumberTitle','off')
    % Extrae columna y agrupa por Algorithm
    vals   = All.(metrics{j});
    groups = All.Algorithm;
    boxplot(vals, groups, 'LabelOrientation','inline','Whisker',1.5)
    ylabel(strrep(metrics{j},'_',' '))
    title(sprintf('Distribución de %s por algoritmo', metrics{j}))
    grid on
end
