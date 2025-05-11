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

%% 3. Agrupa por (gamma, num_trajectories) y calcula el success_rate medio
[G, gammaVals, trajVals] = findgroups(All.gamma, All.num_trajectories);
succMean = splitapply(@mean, All.success_rate, G);

% Construye tabla resumen
Summary = table(gammaVals, trajVals, succMean, ...
                'VariableNames', {'gamma','num_trajectories','succMean'});

%% 4. Pivota para la heat-map
GammaU = unique(Summary.gamma);                   % ejes X
TrajU  = unique(Summary.num_trajectories, 'stable'); % ejes Y
SuccMat = nan(numel(TrajU), numel(GammaU));

for i = 1:height(Summary)
    xi = find(GammaU == Summary.gamma(i));
    yi = find(TrajU  == Summary.num_trajectories(i));
    SuccMat(yi, xi) = Summary.succMean(i);
end

%% 5. Dibuja el heat-map de success-rate
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

%% 6. Gráfico de barras 3D
figure
b = bar3(SuccMat);
colormap(jet)
colorbar
% Añade etiquetas con los valores medios sobre cada barra
for ii = 1:size(SuccMat,1)
    for jj = 1:size(SuccMat,2)
        z = SuccMat(ii,jj);
        if ~isnan(z)
            text(jj, ii, z+0.02, sprintf('%.2f',z), ...
                 'HorizontalAlignment','center','FontSize',8);
        end
    end
end
% Ajusta ticks y etiquetas
xticks(1:numel(GammaList))
xticklabels(arrayfun(@num2str, GammaList, 'UniformOutput',false))
yticks(1:numel(TrajList))
yticklabels(arrayfun(@num2str, TrajList,  'UniformOutput',false))
xlabel('\gamma')
ylabel('# Trayectorias')
zlabel('Success rate medio')
title('Success-rate medio por combinación (barras 3D)')
view(45,30)  % ángulo de cámara
grid on