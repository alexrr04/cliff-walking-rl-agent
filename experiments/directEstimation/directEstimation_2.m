
%% 1. Localiza todos los metrics.csv partiendo de la carpeta “results” al lado de tu script
scriptDir  = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, 'experiment-2');      
files      = dir(fullfile(resultsDir, '**', 'metrics.csv'));

%% 2. Lee y concatena todas las tablas
All = table();
for k = 1:numel(files)
    % Lee el CSV (cada uno contiene 20 filas = 20 runs de esa configuración)
    T = readtable(fullfile(files(k).folder, files(k).name));
    
    % Añade al DataFrame maestro
    All = [All; T];  %#ok<AGROW>
end

%% 3.a Reagrupa solo por training_episodes y calcula media y std
episodes = unique(All.training_episodes);
nEps     = numel(episodes);
succMu   = zeros(nEps,1); succStd = zeros(nEps,1);
rewMu    = zeros(nEps,1); rewStd  = zeros(nEps,1);
stepMu   = zeros(nEps,1); stepStd = zeros(nEps,1);
timeMu   = zeros(nEps,1); timeStd = zeros(nEps,1);

for i = 1:nEps
    e = episodes(i);
    mask = All.training_episodes == e;
    % success rate
    x = All.success_rate(mask);
    succMu(i)  = mean(x); succStd(i) = std(x);
    % reward
    x = All.mean_reward(mask);
    rewMu(i)   = mean(x); rewStd(i)  = std(x);
    % steps
    x = All.mean_steps(mask);
    stepMu(i)  = mean(x); stepStd(i) = std(x);
    % time
    x = All.training_time(mask);
    timeMu(i)  = mean(x); timeStd(i) = std(x);
end

%% 3.b Calcula IC 95% y los imprime
% Asumimos que cada grupo tiene nRuns ejecuciones (aquí suele ser 20)
nRuns = arrayfun(@(e) sum(All.training_episodes==e), episodes);
alpha = 0.05;
tVal  = tinv(1 - alpha/2, nRuns - 1);  % vector de t para cada grupo

% Prealoca vectores de semiancho de IC
succCI = tVal .* succStd  ./ sqrt(nRuns);
rewCI  = tVal .* rewStd   ./ sqrt(nRuns);
stepCI = tVal .* stepStd  ./ sqrt(nRuns);
timeCI = tVal .* timeStd  ./ sqrt(nRuns);

fprintf('  Episodes   Métrica        Media       IC_lower    IC_upper\n');
fprintf('-------------------------------------------------------------\n');
for i = 1:nEps
    e = episodes(i);
    % Success rate
    fprintf('%8d   Success rate  %7.3f   [%6.3f, %6.3f]\n', ...
            e, succMu(i), succMu(i)-succCI(i), succMu(i)+succCI(i));
    % Recompensa
    fprintf('%8d   Rew. media    %7.3f   [%6.3f, %6.3f]\n', ...
            e, rewMu(i),  rewMu(i)-rewCI(i),   rewMu(i)+rewCI(i));
    % Pasos
    fprintf('%8d   Pasos medios  %7.1f   [%6.1f, %6.1f]\n', ...
            e, stepMu(i), stepMu(i)-stepCI(i), stepMu(i)+stepCI(i));
    % Tiempo
    fprintf('%8d   Tiempo (s)    %7.2f   [%6.2f, %6.2f]\n\n', ...
            e, timeMu(i), timeMu(i)-timeCI(i), timeMu(i)+timeCI(i));
end

%% 4. Boxplots
% Convierte 'episodes' a cellstr de etiquetas
epLabels = cellstr(string(episodes));

% Boxplot de Success rate
figure
boxplot(All.success_rate, All.training_episodes, ...
        'Labels', epLabels, ...
        'Whisker', 1.5)
xlabel('Training episodes')
ylabel('Success rate')
title('Distribución de Success rate por nº de episodios')
grid on

% Boxplot de Recompensa media
figure
boxplot(All.mean_reward, All.training_episodes, ...
        'Labels', epLabels, ...
        'Whisker', 1.5)
xlabel('Training episodes')
ylabel('Recompensa media')
title('Distribución de Recompensa media por nº de episodios')
grid on

% Boxplot de Pasos medios
figure
boxplot(All.mean_steps, All.training_episodes, ...
        'Labels', epLabels, ...
        'Whisker', 1.5)
xlabel('Training episodes')
ylabel('Pasos medios')
title('Distribución de Pasos medios por nº de episodios')
grid on

% Boxplot de Tiempo de entrenamiento
figure
boxplot(All.training_time, All.training_episodes, ...
        'Labels', epLabels, ...
        'Whisker', 1.5)
xlabel('Training episodes')
ylabel('Tiempo de entrenamiento (s)')
title('Distribución de Tiempo de entrenamiento por nº de episodios')
grid on