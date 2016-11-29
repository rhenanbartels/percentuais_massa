function batch_p15()
screenSize = get(0, 'ScreenSize');
main_figure = figure('WindowScrollWheelFcn',@moveSlices,...
    'IntegerHandle','off',...
    'Menubar', 'None',...
    'Name', 'Batch P15',...
    'Number', 'Off',...
    'Position', screenSize,...
    'tag', 'main_fig');

main_axis = axes('Parent', main_figure,...
    'tag', 'main_axis');

bckColor = get(main_figure, 'Color');

okButton = uicontrol('Parent', main_figure,...
    'String', 'Apply',...
    'Callback', @applyCallback);

mainMenu = uimenu('Parent', main_figure,...
    'Label', 'File');

uimenu('Parent', mainMenu,...
    'Label', 'Open',...
    'Callback', @openFile);

uimenu('Parent', mainMenu,...
    'Label', 'Open .mat file',...
    'Callback', @openMatFile);

uimenu('Parent', mainMenu,...
    'Label', 'Open Air way',...
    'Callback', @openAirWay);

buttonPosition = get(okButton, 'Position');
airRoiButtonPosition = buttonPosition;
airRoiButtonPosition(1) = airRoiButtonPosition(1) + 80;
tissueRoiButtonPosition = airRoiButtonPosition;
tissueRoiButtonPosition(1) = tissueRoiButtonPosition(1) + 80;
tissueRoiButtonPosition(3) = tissueRoiButtonPosition(1) + 1;
fipButtonPosition = tissueRoiButtonPosition;
fipButtonPosition(1) = tissueRoiButtonPosition(1) + 230;
fipButtonPosition2 = fipButtonPosition;

fipButtonPosition2(1) = fipButtonPosition(1) + 200;

editPosition = airRoiButtonPosition;
editPosition(1) = editPosition(1) + 780;

chkButtonPosition = editPosition;
chkButtonPosition(1) = editPosition(1) + 200;
chkButtonPosition(3) = editPosition(3) + 200;


editSlice = uicontrol('Parent', main_figure,...
    'Style', 'edit',...
    'String', '1',...
    'Position', editPosition,...
    'tag', 'editSlice');

airRoiButton = uicontrol('Parent', main_figure,...
    'Position', airRoiButtonPosition,...
    'String', 'Air Roi',...
    'Callback',@roiAirCallback);

tissueRoiButton = uicontrol('Parent', main_figure,...
    'Position', tissueRoiButtonPosition,...
    'String', 'Tissue Roi',...
    'Callback', @roiTissueCallback);

tissueRoiButton = uicontrol('Parent', main_figure,...
    'Position', fipButtonPosition,...
    'String', 'Flip Image',...
    'Callback', @imageFlip);

tissueRoiButton = uicontrol('Parent', main_figure,...
    'Position', fipButtonPosition2,...
    'String', 'Flip Mask',...
    'Callback', @maskFlip);

tissueRoiButton = uicontrol('Parent', main_figure,...
    'Position', chkButtonPosition,...
    'Style', 'Checkbox',...
    'String', 'Show Mask',...
    'Tag', 'showMask');
 
 uicontrol('Parent', main_figure,...
     'Style', 'text',...
     'Units', 'Normalized',...
     'BackGroundColor', bckColor,...
     'Position', [0.13, 0.95, 0.2, 0.025],...
     'HorizontalAlignment', 'Left',...
     'String', 'Paciente Name:',...
     'Tag', 'txtPatientName')
 
 
  uicontrol('Parent', main_figure,...
     'Style', 'text',...
     'Units', 'Normalized',...
     'BackGroundColor', bckColor,...
     'Position', [0.8, 0.95, 0.15, 0.025],...
     'HorizontalAlignment', 'Left',...
     'String', 'Airway removed: No',...
     'Tag', 'txtRemovedAirWays')

handles = guihandles(main_figure);
guidata(main_figure, handles)
end

function openFile(hObject, eventdata)
     handles = guidata(hObject);
     dirPath = uigetdir('Select Target Folder');     
     if dirPath
         listOfFiles = dir(dirPath);
         fileNames = openDicom(dirPath, listOfFiles);

         %Check if there are DICOMS files in the folder
         if ~isempty(fileNames)
             
             %Get DICOM data
             

             %lung = single(getDicomMatrix(fileNames));
             [lung, ct_info, ct_roi, dirName] = loadCTFromDICOM(0,0,0, fileNames);
             metadata = ct_info{1};
             metadata2 = ct_info{2};
             
             lung = uncalibrateLung(lung, metadata);
             handles.dicom.rawLung = lung;
             handles.dicom.lung = lung;
             
                                   
             [maskFileName, maskPath] = uigetfile('*.nrrd;*.hdr','Select PARENCHYMA Mask', dirPath);
             
             if maskFileName     
                 
                 sliceNumber = str2double(get(handles.editSlice, 'string'));
                 
                 masks = flipdim(openMask([maskPath maskFileName], handles), 3);    
                 handles.dicom.mask = masks;
                 handles.dicom.dirPath = dirPath;
                 handles.dicom.metadata = metadata;
                 handles.dicom.metadata2 = metadata2;
                 guidata(hObject, handles);                 
                 displayLung(handles, sliceNumber)
                 
                 setPatientName(handles, metadata)
                 
                 guidata(hObject, handles);
                 set(handles.txtRemovedAirWays, 'String', sprintf('Airway Removed: %s', 'No'))
             else
                 warndlg('Operation Canceled!', 'Canceled')
             end
         else
             warndlg('No DICOMS file was found!', 'No DICOM')
         end

     end
end

function openMatFile(hObject, eventdata)
    handles = guidata(hObject);
    [fileName, dirPath] = uigetfile('*.mat', 'Selecione arquivo mat',...
    '/Users/lismariecarvalho/Documents/ProcessedData_SDRA/CT_1Month');
    if fileName
        data = load([dirPath, fileName]);
        metadata = data.Dicom.Info;
        metadata2 = data.Dicom.Info;
        
        handles.dicom.rawLung = flipdim(uncalibrateLung(single(data.Dicom.Image), metadata), 3);
        handles.dicom.lung = flipdim(uncalibrateLung(single(data.Dicom.Lung), metadata), 3);
        handles.dicom.mask = flipdim(data.Dicom.Masks, 3);
        handles.dicom.dirPath = dirPath;
        handles.dicom.metadata = metadata;
        handles.dicom.metadata2 = metadata2;
        guidata(hObject, handles);
        sliceNumber = str2double(get(handles.editSlice, 'string'));
        displayLung(handles, sliceNumber)
        
        setPatientName(handles, metadata)
        
        guidata(hObject, handles);
        
    end
end

function setPatientName(handles, metadata)
    msgHolder = 'Patient Name: %s';
    set(handles.txtPatientName, 'String', sprintf(msgHolder,...
         metadata.PatientName.FamilyName))
end

function fileNames = openDicom(dirPath, listOfFiles)
fileNames = {};
counter = 1;

nFiles = length(listOfFiles);

h = waitbar(0, 'Getting folder information...');
for index = 1:nFiles
    currentFileName = listOfFiles(index).name;
    if ~strcmp(currentFileName, '.') && ~strcmp(currentFileName, '..')
        filePath = [dirPath filesep currentFileName];
        try
            dicominfo(filePath);
            fileNames{counter} = filePath;
            counter = counter + 1;
        catch
        end
    end
    waitbar(index / nFiles)
end
close(h)
end

function masks = openMask(fileName, handles)

    if strfind(fileName,'hdr')
        masks = analyze75read(fileName);
    else
        masks = nrrd_read(fileName);
        masks(masks > 500) = 0;
        set(handles.txtRemovedAirWays, 'String',...
            sprintf('Airway Removed: %s', 'Yes'))
    end
    
end

function metadata = getDicomInformation(fileName)
    metadata = dicominfo(fileName);
end

function dicomMatrix = getDicomMatrix(fileNames)   
    h = waitbar(0, 'Reading DICOM files...');
    counter =  1;
    
    nFiles = length(fileNames); 
    for index = 1:nFiles
        if isempty(strfind(fileNames{index}, 'bak'))
            try
                dicomMatrix(:, :, counter) = dicomread(fileNames{index});
                counter = counter + 1;
            catch
            end            
            waitbar(counter / nFiles)
        end
    end
    close(h)
end

function lung = uncalibrateLung(lung, metadata)
     lung = (lung - metadata.RescaleIntercept) / metadata.RescaleSlope;    
end
 
function imageFlip(hObject, eventdata)
    handles = guidata(hObject);
    handles.dicom.rawLung = flipdim(handles.dicom.rawLung, 3);    
    value = str2double(get(handles.editSlice, 'string'));
    displayLung(handles, value)
    guidata(hObject, handles)
    
end

 
function maskFlip(hObject, eventdata)
    handles = guidata(hObject);
    handles.dicom.mask = flipdim(handles.dicom.mask, 3);
    guidata(hObject, handles)   
end


function displayLung(handles, sliceNumber)

imagesc(handles.dicom.rawLung(:, :, sliceNumber));
colormap(gray(256));axis equal;axis off

end

function roiAirCallback(hObject, eventdata)
handles = guidata(hObject);
[handles.roi_air_properties.handle, handles.roi_air_properties.position] =...
    drag_and_drop(100, 100, 8, 'g'); 
guidata(hObject, handles)
end

function roiTissueCallback(hObject, eventdata)
handles = guidata(hObject);
[handles.roi_tissue_properties.handle, handles.roi_tissue_properties.position] =...
    drag_and_drop(100, 100, 8, 'r'); 
guidata(hObject, handles)
end

function moveSlices(hObject, eventdata)
handles = guidata(hObject);
value = str2double(get(handles.editSlice, 'string'));
if eventdata.VerticalScrollCount > 0
    value = value + 1;
else
    value = value - 1;
end

[x, y, z] = size(handles.dicom.lung);

if value > 0 && value < z
    set(handles.editSlice, 'string', num2str(value))
    displayLung(handles, value)
    isToShowMask = get(handles.showMask, 'Value');
    if isToShowMask
        showMask(handles.dicom.lung(:, :, value),...
            handles.dicom.mask(:, :, value))
    end
    
end

end

function showMask(lung, mask)

     mask = mask >= 1;
     poly = mask2poly(mask, 'Inner');
% 
%     color1 = 0; color2 = 0.8; color3 = 0;
%     colorMask = cat(3, color1 * ones(size(lung)), color2 * ones(size(lung)),...
%         color3 * ones(size(lung)));
%     
%     hold on
%     h = imshow(colorMask);
%     set(h, 'AlphaData', mask);
    hold on 
    plot(poly(:, 1), poly(:, 2), 'g')
    
end


function rawLung = lungCalibration(rawLung, mAir, mTissue)
coef = polyfit([mAir, mTissue], [-1000 50], 1);
rawLung = rawLung * coef(1) + coef(2);
end

function [massPerDensity, volumePerDensity, huValues] =...
    lungAnalysis(handles, rawLung, mask, metadata, metadata2, roiAir, roiTissue)
voxelVolume = calculateVoxelVolume(metadata, metadata2);

roiAir = -1000;
roiTissue = 50;

rawLung(mask == 0) = 10000;



rawLung = single(int16(rawLung));

sliceNumber = str2double(get(handles.editSlice, 'string'));

figure;imagesc(rawLung(:, :, sliceNumber))

rawLung(rawLung < -1000 | rawLung > 100) = [];



huValues = unique(rawLung);
nLung = length(rawLung);

massPerDensity = zeros(1, length(huValues));
volumePerDensity = zeros(1, length(huValues));
voxelPerDensity = zeros(1, length(huValues));
counter = 1;

nVoxels = length(huValues);

%w = waitbar(0, 'Calculating...');

for hu = huValues    
    nVoxels = length(rawLung(rawLung == hu));
    massPerDensity(counter) = ((hu - roiAir)/(roiTissue - roiAir))...
        * voxelVolume * 1.04 * nVoxels;
    volumePerDensity(counter) = nVoxels * voxelVolume / 1000;
    voxelPerDensity(counter) = nVoxels / nLung * 100;
    rawLung(rawLung == hu) = [];
    counter = counter + 1;
    %waitbar(counter / length(huValues))
end
%close(w)
sumMass = cumsum(massPerDensity);

sumVolume = cumsum(volumePerDensity);
cumulativeVolume = sumVolume / sumVolume(end) * 100;
cumulativeVoxel = cumsum(voxelPerDensity);

[~, pos] = min(abs(cumulativeVolume - 15));
[~, pos3] = min(abs(cumulativeVolume - 3));
[~, pos85] = min(abs(cumulativeVolume - 85));
[~, pos97] = min(abs(cumulativeVolume - 97));

p15Mass = sumMass(pos);
p3Mass = sumMass(pos3);
p85Mass = sumMass(end) - sumMass(pos85);
p97Mass = sumMass(end) - sumMass(pos97);

p15Volume = sumVolume(pos);
p3Volume = sumVolume(pos3);
p85Volume = sumVolume(end) - sumVolume(pos85);
p97Volume = sumVolume(end) - sumVolume(pos97);


p15Density = p15Mass / p15Volume;
p3Density = p3Mass / p3Volume;
p85Density = p85Mass / p85Volume;
p97Density = p97Mass / p97Volume;

totalVolume = sumVolume(end);
totalMass = sumMass(end);

disp('Done!')
end


function voxelVolume = calculateVoxelVolume(metadata, metadata2)
if isfield(metadata,'SpacingBetweenSlices');
    if isfield(metadata,'SliceThickness')
        if abs(metadata.SpacingBetweenSlices) < metadata.SliceThickness
            voxelVolume =(metadata.PixelSpacing(1) ^ 2 *...
                metadata.SliceThickness * 0.001) *...
                (abs(metadata.SpacingBetweenSlices) / metadata.SliceThickness);
        else
            voxelVolume = (metadata.PixelSpacing(1) ^ 2 *...
                metadata.SliceThickness * 0.001);
        end
    else
        voxelVolume = (metadata.PixelSpacing(1) ^ 2 *...
            metadata.SliceThickness * 0.001);
    end
else
    
    if isfield(metadata,'SliceThickness')==1;
        thick=abs(metadata.SliceThickness);
    elseif isfield(metadata,'SpacingBetweenSlices');
        thick=abs(metadata.SpacingBetweenSlices);
    else
        thick=abs(metadata.PixelDimensions(3));
    end
    
    SpacingBetweenSlices = abs(metadata2.SliceLocation - metadata.SliceLocation);

    SliceThickness = metadata.SliceThickness;
    voxelVolume = (metadata.PixelSpacing(1) ^ 2 * thick * 0.001) * (SpacingBetweenSlices / SliceThickness);
end
end

function showMaskOverLung(lung, mask)
fig = figure;
intensity = 300;
maskM3 = mask * intensity;
c1M3=0.5; c2M3=0.8; c3M3=0.2;
maskColorM3 = cat(3, c1M3 * ones(size(lung)), c2M3 * ones(size(lung)), c3M3 * ones(size(lung)));
imshow(lung)
%colormap(gray)
hold on
h = imshow(maskColorM3);
hold off
set(h, 'AlphaData', maskM3 * 0.3)
uiwait(fig)
end

function applyCallback(hObject, eventdata)
handles = guidata(hObject);
dirPath = handles.dicom.dirPath;
showMaskOverLung(handles.dicom.rawLung(:, :, 70), handles.dicom.mask(:,:, 70))
[mAir, roiAir] = averageCircle(hObject, eventdata, 'air');
[mTissue, roiTissue] = averageCircle(hObject, eventdata, 'tissue');
rawLung = lungCalibration(handles.dicom.rawLung, mAir, mTissue);
%Discover where the mask
maskPosition = find(sum(sum(handles.dicom.mask)) >= 1);
maskTotal = handles.dicom.mask(:, :, maskPosition);
rawLung = rawLung(:, :, maskPosition);

y = unique(maskTotal);
ny = length(y);
nSlices = size(rawLung, 3);

maskTotal = prepareMask(maskTotal, ny, [], 0);


finalPathIndex = strfind(dirPath, filesep);
%finalPath = dirPath(1:finalPathIndex(end) - 1);

[massPerDensityTotal, volumePerDensityTotal, huValuesTotal] = lungAnalysis(handles, rawLung,...
    maskTotal, handles.dicom.metadata, handles.dicom.metadata2, mAir, mTissue);



[m3Total, m15Total, m85Total, m97Total, p3Total, p15Total, p85Total, p97Total] = calculateMx(volumePerDensityTotal,...
    massPerDensityTotal);


results.perDensity.massPerDensityTotal = massPerDensityTotal;
results.perDensity.volumePerDensityTotal = volumePerDensityTotal;

results.roiTissue = roiTissue;
results.roiAir = roiAir;

maskInf =  prepareMask(maskTotal, ny, nSlices, 11);
maskMedium = prepareMask(maskTotal, ny, nSlices, 10);
maskTop = prepareMask(maskTotal, ny, nSlices, 9);

[massPerDensityInf, volumePerDensityInf, huValuesInf] = lungAnalysis(handles, rawLung,...
    maskInf, handles.dicom.metadata, handles.dicom.metadata2, mAir, mTissue);


[m3Inf, m15Inf, m85Inf, m97Inf, p3Inf, p15Inf, p85Inf, p97Inf] = calculateMx(volumePerDensityInf,...
    massPerDensityInf);


[massPerDensityMed, volumePerDensityMed, huValuesMed] = lungAnalysis(handles, rawLung,...
    maskMedium, handles.dicom.metadata, handles.dicom.metadata2, mAir, mTissue);


[m3Med, m15Med, m85Med, m97Med, p3Med, p15Med, p85Med, p97Med] = calculateMx(volumePerDensityMed,...
    massPerDensityMed);


[massPerDensityTop, volumePerDensityTop, huValuesTop] = lungAnalysis(handles, rawLung,...
    maskTop, handles.dicom.metadata, handles.dicom.metadata2, mAir, mTissue);


[m3Top, m15Top, m85Top, m97Top, p3Top, p15Top, p85Top, p97Top] = calculateMx(volumePerDensityTop,...
    massPerDensityTop);


results.perDensity.massPerDensityInf = massPerDensityInf;
results.perDensity.volumePerDensityInf = volumePerDensityInf;


results.perDensity.massPerDensityMed = massPerDensityMed;
results.perDensity.volumePerDensityMed = volumePerDensityMed;

results.perDensity.massPerDensityTop = massPerDensityTop;
results.perDensity.volumePerDensityTop = volumePerDensityTop;

%results.dicom.rawLung = handles.dicom.rawLung;
%results.dicom.mask = handles.dicom.mask;
%results.dicom.roiAir = roiAir;
%results.dicom.roiTissue = roiTissue;
results.dicom.huValuesTotal = huValuesTotal;
results.dicom.huValuesInf = huValuesInf;
results.dicom.huValuesMed = huValuesMed;
results.dicom.huValuesTop = huValuesTop;

%Mass Results
%Whole lung
results.mass.m3Total = m3Total;
results.mass.m15Total = m15Total;
results.mass.m85Total = m85Total;
results.mass.m97Total = m97Total;

%Inf lung
results.mass.m3Inf = m3Inf;
results.mass.m15Inf = m15Inf;
results.mass.m85Inf = m85Inf;
results.mass.m97Inf = m97Inf;

%Med lung
results.mass.m3Med = m3Med;
results.mass.m15Med = m15Med;
results.mass.m85Med = m85Med;
results.mass.m97Med = m97Med;

%Top lung
results.mass.m3Top = m3Top;
results.mass.m15Top = m15Top;
results.mass.m85Top = m85Top;
results.mass.m97Top = m97Top;


%Density Results
%Whole lung
results.density.p3Total = p3Total;
results.density.p15Total = p15Total;
results.density.p85Total = p85Total;
results.density.p97Total = p97Total;

%Inf lung
results.density.p3Inf = p3Inf;
results.density.p15Inf = p15Inf;
results.density.p85Inf = p85Inf;
results.density.p97Inf = p97Inf;

%Med lung
results.density.p3Med = p3Med;
results.density.p15Med = p15Med;
results.density.p85Med = p85Med;
results.density.p97Med = p97Med;

%Top lung
results.density.p3Top = p3Top;
results.density.p15Top = p15Top;
results.density.p85Top = p85Top;
results.density.p97Top = p97Top;

results.info =  handles.dicom.metadata;

saveResults(results)

disp(handles.dicom.metadata.PatientName.FamilyName)
msgbox('Done!', 'Done!')
end


function saveResults(results)
    [fileName, pathName] = uiputfile('*.mat');
    if fileName
       save([pathName fileName], 'results')
    end
end

function mask = prepareMask(mask, ny, nSlices, index)
    
if ny > 2
    mask(mask > 500) = 0;
    if nargin == 4 && index
        mask(mask ~= index & mask ~= index + 3) = 0;
    end
else
    if nargin == 4 && index
            [firstLevel, secondLevel, thirdLevel] = lungThresholds(nSlices);
        if index == 11;
            mask(:, :, 1 : secondLevel(2)) = 0;
        elseif index == 10
            mask(:, :, 1 : firstLevel(2)) = 0;
            mask(:, :, secondLevel(end) + 1 : thirdLevel(2)) = 0;
        else
            mask(:, :, secondLevel(1) : thirdLevel(end)) = 0;
        end
        
    end
end
end

function [firstLevel, secondLevel, thirdLevel] = lungThresholds(nSlices)
    %Centimeters lung size;
    oneThird = nSlices / 3;
    floorOneThird = floor(oneThird);

    diffOneThird = abs(floorOneThird - oneThird);

    if (diffOneThird)
        firstLevel = [1, floorOneThird + 1];
        if diffOneThird > 0 && diffOneThird < 0.5
            secondLevel = [firstLevel(2) + 1, firstLevel(2) + floorOneThird];
        else
            secondLevel = [firstLevel(2) + 1, firstLevel(2) + floorOneThird + 1];
        end

    else
        firstLevel = [1 floorOneThird];
        secondLevel = [floorOneThird + 1, firstLevel(2) + floorOneThird];
    end

    thirdLevel = [secondLevel(2) + 1, secondLevel(2) + floorOneThird];
    assert(thirdLevel(2) == nSlices, 'Foi diferente')
end

function [m3, m15, m85, m97, p3, p15, p85, p97] = calculateMx(volumePerDensity, massPerDensity)
    cumsumVolume  = cumsum(volumePerDensity);
    volumePerc = cumsumVolume / cumsumVolume(end) * 100;
    cumsumMass = cumsum(massPerDensity);
    
    [trash, index3] = min(abs(volumePerc  - 3));
    [trash, index15] = min(abs(volumePerc  - 15));
    [trash, index85] = min(abs(volumePerc  - 85));
    [trash, index97] = min(abs(volumePerc  - 97));
    
    m3 = cumsumMass(index3);
    m15 = cumsumMass(index15);
    m85 = cumsumMass(end) - cumsumMass(index85);
    m97 = cumsumMass(end) - cumsumMass(index97);
    
    p3 = m3 / cumsumVolume(index3);
    p15 = m15 / cumsumVolume(index15);
    p85 = m85 / (cumsumVolume(end) - cumsumVolume(index85));
    p97 = m97 / (cumsumVolume(end) - cumsumVolume(index97));
    
end

function [m, imgMask] = averageCircle(hObject, eventdata, roi_type)
%meanDisk computes mean of values inside a circle
%   M = meanDisk(IMG, XC, YC, R) returns the mean of IMG(Y,X) for all X and
%   Y such that the Euclidean distance of (X,Y) from (XC,YC) is less than
%   R. IMG must be 2-D, R must be positive, and some elements of IMG must
%   lie within the circle.
% This section is for efficiency only - avoids wasting computation time on
% pixels outside the bounding square
handles = guidata(hObject);
switch roi_type
    case 'air'
        circle_object = findobj(gca, 'Type', 'rectangle','-and',...
            'EdgeColor', 'g');
        roi_handle = handles.roi_air_properties;        
    otherwise
        circle_object = findobj(gca, 'Type', 'rectangle','-and',...
            'EdgeColor', 'r');
        roi_handle = handles.roi_tissue_properties;
end
position = get(circle_object, 'Position');
%x do circulo
r = position(3) / 2;
xc = position(1) + r; %reposiciona o centro do circulo.
%y do circulo
yc = position(2) + r; %reposiciona o centro do circulo.
%raio

img = handles.dicom.rawLung;

slice = str2double(get(handles.editSlice, 'string'));
[sy sx] = size(handles.dicom.rawLung(:,:, slice));
xmin = max(1, floor(xc-r));
xmax = min(sx, ceil(xc+r));
ymin = max(1, floor(yc-r));
ymax = min(sy, ceil(yc+r));
img = img(ymin:ymax, xmin:xmax, slice); % trim boundaries
%figure;imagesc(img);colormap(gray(156))
xc = xc - xmin + 1;
yc = yc - ymin + 1;
% Make a circle mask
[x y] = meshgrid(1:size(img,2), 1:size(img,1));
mask = (x-xc).^2 + (y-yc).^2 < r.^2;
% Compute mean
m = sum(sum(double(img) .* mask)) / sum(mask(:));
imgMask = double(img) .* mask;
end


function [X] = nrrd_read(filename)
% Modified from the below function (only X is outputed and cleaner is not used)
%NRRDREAD  Import NRRD imagery and metadata.
%   [X, META] = NRRDREAD(FILENAME) reads the image volume and associated
%   metadata from the NRRD-format file specified by FILENAME.
%
%   Current limitations/caveats:
%   * "Block" datatype is not supported.
%   * Only tested with "gzip" and "raw" file encodings.
%   * Very limited testing on actual files.
%   * I only spent a couple minutes reading the NRRD spec.
%
%   See the format specification online:
%   http://teem.sourceforge.net/nrrd/format.html

% Copyright 2012 The MathWorks, Inc.

try
    % Open file.
    fid = fopen(filename, 'rb');
    assert(fid > 0, 'Could not open file.');
%     cleaner = onCleanup(@() fclose(fid));

    % Magic line.
    theLine = fgetl(fid);
    assert(numel(theLine) >= 4, 'Bad signature in file.')
    assert(isequal(theLine(1:4), 'NRRD'), 'Bad signature in file.')

    % The general format of a NRRD file (with attached header) is:
    % 
    %     NRRD000X
    %     <field>: <desc>
    %     <field>: <desc>
    %     # <comment>
    %     ...
    %     <field>: <desc>
    %     <key>:=<value>
    %     <key>:=<value>
    %     <key>:=<value>
    %     # <comment>
    % 
    %     <data><data><data><data><data><data>...

    meta = struct([]);

    % Parse the file a line at a time.
    while (true)

      theLine = fgetl(fid);

      if (isempty(theLine) || feof(fid))
        % End of the header.
        break;
      end

      if (isequal(theLine(1), '#'))
          % Comment line.
          continue;
      end

      % "fieldname:= value" or "fieldname: value" or "fieldname:value"
      parsedLine = regexp(theLine, ':=?\s*', 'split','once');

      assert(numel(parsedLine) == 2, 'Parsing error')

      field = lower(parsedLine{1});
      value = parsedLine{2};

      field(isspace(field)) = '';
      meta(1).(field) = value;

    end

    datatype = getDatatype(meta.type);

    % Get the size of the data.
    assert(isfield(meta, 'sizes') && ...
           isfield(meta, 'dimension') && ...
           isfield(meta, 'encoding') && ...
           isfield(meta, 'endian'), ...
           'Missing required metadata fields.')

    dims = sscanf(meta.sizes, '%d');
    ndims = sscanf(meta.dimension, '%d');
    assert(numel(dims) == ndims);

    data = readData(fid, meta, datatype);
    data = adjustEndian(data, meta);

    % Reshape and get into MATLAB's order.
    X = reshape(data, dims');
    X = permute(X, [2 1 3]);
    
    fclose(fid);
catch err
    fclose(fid);
    rethrow(err)
end
end

function datatype = getDatatype(metaType)

% Determine the datatype
switch (metaType)
 case {'signed char', 'int8', 'int8_t'}
  datatype = 'int8';
  
 case {'uchar', 'unsigned char', 'uint8', 'uint8_t'}
  datatype = 'uint8';

 case {'short', 'short int', 'signed short', 'signed short int', ...
       'int16', 'int16_t'}
  datatype = 'int16';
  
 case {'ushort', 'unsigned short', 'unsigned short int', 'uint16', ...
       'uint16_t'}
  datatype = 'uint16';
  
 case {'int', 'signed int', 'int32', 'int32_t'}
  datatype = 'int32';
  
 case {'uint', 'unsigned int', 'uint32', 'uint32_t'}
  datatype = 'uint32';
  
 case {'longlong', 'long long', 'long long int', 'signed long long', ...
       'signed long long int', 'int64', 'int64_t'}
  datatype = 'int64';
  
 case {'ulonglong', 'unsigned long long', 'unsigned long long int', ...
       'uint64', 'uint64_t'}
  datatype = 'uint64';
  
 case {'float'}
  datatype = 'single';
  
 case {'double'}
  datatype = 'double';
  
 otherwise
  assert(false, 'Unknown datatype')
end
end


function data = readData(fidIn, meta, datatype)

switch (meta.encoding)
 case {'raw'}
  
  data = fread(fidIn, inf, [datatype '=>' datatype]);
  
 case {'gzip', 'gz'}

  tmpBase = tempname();
  tmpFile = [tmpBase '.gz'];
  fidTmp = fopen(tmpFile, 'wb');
  assert(fidTmp > 3, 'Could not open temporary file for GZIP decompression')
  
  tmp = fread(fidIn, inf, 'uint8=>uint8');
  fwrite(fidTmp, tmp, 'uint8');
  fclose(fidTmp);
  
  gunzip(tmpFile)
  
  fidTmp = fopen(tmpBase, 'rb');
%   cleaner = onCleanup(@() fclose(fidTmp));
  
  meta.encoding = 'raw';
  data = readData(fidTmp, meta, datatype);
  fclose(fidTmp)
  
 case {'txt', 'text', 'ascii'}
  
  data = fscanf(fidIn, '%f');
  data = cast(data, datatype);
  
 otherwise
  assert(false, 'Unsupported encoding')
end
end


function data = adjustEndian(data, meta)

[void,void,endian] = computer();

needToSwap = (isequal(endian, 'B') && isequal(lower(meta.endian), 'little')) || ...
             (isequal(endian, 'L') && isequal(lower(meta.endian), 'big'));
         
if (needToSwap)
    data = swapbytes(data);
end
end

function openAirWay(hObject, eventdata)
   handles = guidata(hObject);
   [fileName pathName] = uigetfile('*.nrrd','', handles.dicom.dirPath);
   
   if fileName
       try
           airWayMask = flipdim(openMask([pathName fileName], handles), 3);
           handles.dicom.mask(airWayMask ~= 0) = 0;
           set(handles.txtRemovedAirWays, 'String', sprintf('Airway Removed: %s', 'Yes'))
           guidata(hObject, handles)
       catch err
           errordlg(err.message)
       end
   end
end


function [ct, ct_info, ct_roi, dirName] = loadCTFromDICOM(selectFolder, dirName,...
    warningOverlap, fileNames)
    
    if nargin < 3
        warningOverlap = true;
    end

    % If required, let the user select the directory graphically.
    % Else, the user provides the directory name explicitly.
    if selectFolder,
        dirName = uigetdir;
    end
    
    ct = [];
    ct_info = {};
    ct_roi = {};
    
    files = fileNames;
    poly_files = dir(fullfile(dirName,'*_poly.mat'));
    poly_count = 1;
    
    numberFiles = length(fileNames);
    
    if numberFiles==0,
        %display(sprintf('loadCTFromDICOM: No DICOM files found in directory %s',...
        %                dirName));
        return;
    end
    
    ct_1 = dicominfo(files{1});
    
    use_slice_location = true;
    sliceLocations = zeros(1,numberFiles);
    
   for k=1:numberFiles,
        ct_bla = dicominfo(files{k});
        if ~isfield(ct_bla, 'SliceLocation'),
            %display('loadCTFromDICOM: Error - File has no field SliceLocation. Nothing is loaded.');
            %return;
            %display('loadCTFromDICOM: Error - File has no field SliceLocation.');
            use_slice_location = false;
            break;
        end
        ct_i = dicominfo(files{k});
        sliceLocations(k) = ct_i.SliceLocation;
    end
    
    if use_slice_location
        last_pos = ct_1.SliceLocation;
        new_pos = 0;
        [~, I] = sort(sliceLocations,'ascend');
        files = files(I);
    end
    
     h = waitbar(0, 'Loading Dicoms...');
    
    for k=1:numberFiles,
        ct(:,:,k) = dicomread(files{k});
        ct_i = dicominfo(files{k});
        ct_info{k} = ct_i;
        
        if use_slice_location
            if k>1
                new_pos = ct_i.SliceLocation;
                if warningOverlap
                    if abs(new_pos - last_pos) ~= ct_i.SliceThickness
                        %display(sprintf('loadCTFromDICOM: Slice overlap detected between slices %d and %d (%d mm, SliceThickness = %d)',...
                                        %k-1, k, abs(new_pos - last_pos), ct_i.SliceThickness));
                    end
                end
            end
            last_pos = ct_i.SliceLocation;
        end
        
        if isfield(ct_i, 'Private_6001_10c0')
            roi_info = ct_i.Private_6001_10c0.Item_1;
            count = 1;
            fields = fieldnames(roi_info);
            for l=1:length(fields),
                f = fields{l};
                if ~isempty(strfind(f,'_10b0'))
                    if strcmp(roi_info.(f),'POLYGON')
                        ind = textscan(f,'%s','delimiter','_');
                        poly_roi = roi_info.(strcat(ind{1}{1},'_',ind{1}{2},'_10ba'));
                        ct_roi{k}{count} = [poly_roi(1:2:end-1) poly_roi(2:2:end-1)];
                        count = count+1;
                    end
                end
            end
            if count==1
                ct_roi{k} = {};
            end
            
        else
            fn_poly = strcat(dirName,'/',strtok(files{k},'.'),'_poly.mat');
            if exist(fn_poly,'file')
                S = load(fn_poly);
                ct_roi{k} = S.slicePoly;
            else
                ct_roi{k} = {};
                %display(strcat('No ROI information in slice ', int2str(k)));
            end
        end
        waitbar(k / numberFiles)
    end
    close(h)
    
    ct = int16(ct)*ct_info{1}.RescaleSlope + ct_info{1}.RescaleIntercept;
    
    %display(sprintf('loadCTFromDICOM: Succesfully loaded %d slices from directory %s',...
    %                numberFiles, dirName));
end


function poly=mask2poly(mask,countourType,sortPointsMethod)
%% function poly=mask2poly(mask)
% Finds a polygon enclosing the user defind mask of logicals. Kind of a
%  reverse/complementary of Matlab poly2mask function.
%
%% Syntax
% poly=mask2poly(mask);
% poly=mask2poly(mask,countourType);
% poly=mask2poly(mask,countourType,sortPointsMethod);
%
%% Description
% This functions goal is to find a poligon which enclosures a user supplied mask. It's a
%  kind of a complementary of Matlab poly2mask function. The difference is that all
%  contour points are returned- wihout missing points for linearly related points. In
%  order to get a 100% complementary of poly2mask all points inside straight lines shoud
%  be ommited. In my case I actually need all those points, as indexs of ROI. 
%  Combinng mask2poly with poly2mask the user can produce a mask from a contour (line with
%  X and Y coordinates), and vise-versa.
%
%% Input arguments:
% mask- two dimentional matrix of numbers (all numeric types are supported, though mask is
%  usally a matix of logicals).
%
% countourType- {['Inner'],'Outer','Exact'} a string describing the desired contour type.
%  'Inner' (default) will result in a contour inside the mask- the largest shape included
%     by the mask.
%  'Outer' will result in a contour ouside the mask- the smallest shape including by the
%     mask.
%  'Exact' option will result in a contour between 'Inner' and 'Outer', and it lies
%     exactly on the mask margins. 
% Both 'Outer' and 'Inner' results are integers, ready to be used for indexing.
% so it can be used for indexing, as oposed to 'Exact' results which are doubles, and
% cannot be used for indexing.
%
% sortPointsMethod- two methds are currently implemented:
%  'CW'- Clock Wise- an efficinet and fast, but can create "saw tooth" shaped contour.
%  'MINDIST'- minimal siatnce between points- will usally result in a better contour,
%     without "saw tooth" shaped contour. but the price to pay is ~X20 times slower run
%     time.
% other value of sortPointsMethod will skip sorting points.
%
%% Output arguments
% poly- Two dimentional [N,2] matirx with coordinates of all points of the contour. Each
%  point is described by an appropriate row. X is described by the first column, Y by the
%  second.
%
%% Issues & Comments (None)
%
%% Example
% x = [76    51    82    97   118   167   180   145   113    92  76];
% y = [41    73   115    80   143   173   120    57    40    33  41];
% mask = poly2mask(x,y,200,200);
% figure;
% imshow(mask);
% hold on;
% poly=mask2poly(mask,'Inner','CW');
% plot(poly(:,1),poly(:,2),'v-g','MarkerSize',9,'LineWidth',4);
% poly=mask2poly(mask,'Inner','MinDist');
% plot(poly(:,1),poly(:,2),'s-k','MarkerSize',12,'LineWidth',2);
% poly=mask2poly(mask,'Outer');
% plot(poly(:,1),poly(:,2),'*m','MarkerSize',9);
% poly=mask2poly(mask,'Exact');
% plot(poly(:,1),poly(:,2),'.r','MarkerSize',18);
% 
% plot(x,y,'O-b','MarkerSize',12,'LineWidth',3);
% hold off;
% legend('mask2poly- Inner- CCW','mask2poly- Inner- MinDist','mask2poly- Outer','mask2poly- Exact','poly2mask');
% title('mask2poly Vs. poly2mask','FontSize',14);
%
%% See also
% poly2mask;            % Matlab function
% imrect;               % Matlab function
% imroi;                % Matlab function
% sortPoint2ContourCW   % Custom function
% sortPointMinDist      % Custom function
%
%% Revision history
% First version: Nikolay S. 2011-07-07.
% Last update:   Nikolay S. 2011-07-25.
%
% *List of Changes:*
%   ------------------2011-07-25-------------------------
% - Ordering points accourding to rule of "nearest point" (acurate but slow) added.
%   ------------------2011-07-14-------------------------
% - An option to reorder the points so it will define a CW contour.
%   ------------------2011-07-13-------------------------
% - "Inner" and "Outer" options replaced isIndex option
% - Diff based edges calculation replaced contour based calculation for "Inner" and
%  "Outer" options, which resulted in ~x3 shorter run time.
% 

if nargin<3
   sortPointsMethod='None';
   if nargin<2
      countourType='Inner'; %{'Inner','Outer','Exact'}
   end
end

%% Pad mask to deal with edges on contours
paddedMask=false(2+size(mask));
paddedMask(1+(1:size(mask,1)),1+(1:size(mask,2)),:)=mask;
doubleMask=double(paddedMask);
countourType=upper(countourType);

switch (countourType)
   case({'INNER','OUTER'})
      %% Caculate via Gradient fast but up-to indesx exact 
      maskEdges=abs(doubleMask-circshift(doubleMask,[1,0,0]))+...
         abs(doubleMask-circshift(doubleMask,[0,1,0]))+...
         abs(doubleMask-circshift(doubleMask,[-1,0,0]))+...
         abs(doubleMask-circshift(doubleMask,[0,-1,0]));
      if strcmpi(countourType,'OUTER')
         paddedMask=~paddedMask; % Outer edges mark
      end
      [edgeRows,edgeCols]=find(maskEdges>0 & paddedMask);
      maskContours=cat(2,edgeCols,edgeRows); 
      
      switch(upper(sortPointsMethod))
         case('CW')
            [xCW,yCW]=sortPoint2ContourCW(maskContours(:,1),maskContours(:,2));
            maskContours=cat(2,xCW,yCW);
         case('MINDIST')
            [xCW,yCW]=sortPointMinDist(maskContours(:,1),maskContours(:,2));
            maskContours=cat(2,xCW,yCW);
      end % switch(upper(sortPointsMethod))
      
   otherwise
      %% Caculate via contour- slow yet accurate and easy to implement
      contourTresh=0.5*max(doubleMask(:));
      maskContours=contourc(doubleMask,[contourTresh,contourTresh]);
      maskContours=transpose(maskContours); % Convert to standart Pos coordinates system
end

%% Fix the inacurities caused by padding
maskContours=maskContours-1;

poly=maskContours;
end
