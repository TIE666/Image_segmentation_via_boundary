function [seg] = segment_image(I)
    % Segment an image combing different methods.
    % Author: 
    %   Yuhao Li (2020.11)
    % Input: 
    %   I: 3D matrix in double storing a RGB colour image.
    % Output: 
    %   seg: 2D matrix with same shape to I, a binary image where 
    %        values of 1 indicate the locations of boundaries between 
    %        different regions of the image.
    % Layout:
    %   complex cells -----------> edges(threshold) -
    %                                                |
    %   2-means clustering ------> edges(bwperim) -----  
    %                                                |
    %   Entropy texture filter --> edges(Canny) -----  
    %                                                |
    %   Region growing ----------> edges(Canny) -----
    %                                                |
    %   Refering each others, to guess the correct boundary
    %   
    %   Using multiple different methods, trys to segment the image
    %   with respect to different features. For example, complex cells
    %   method extracts bare, strong edges, clustering extracts edges that 
    %   separate colours, entropy filter extracts edges that separate
    %   textures. Finally, letting edges extracted from one method refer
    %   back to those from different methods, in order to remove the
    %   meaningless ones and keep those which might indicate contours.
    %  
    % Acknowledgement:
    %   This code uses methods that are implemented by someone else.
    %   - gabor2.m: 
    %       This is given in KCL 7CCSMCVI & 6CCS3COV 2020-2021 coursework 4.
    %   - regiongrowing.m:
    %       This is an implementation of region growing for a single 
    %       region, from matlab file exchange. 
    %       Dirk-Jan Kroon (2020). 
    %       Region Growing (https://www.mathworks.com/matlabcentral/fileexchange/19084-region-growing), 
    %       MATLAB Central File Exchange. Retrieved November 11, 2020.
    
    maxR = size(I,1);
    maxC = size(I,2);
    
    % Complex cells
    ccI1 = complex_cells(I(:,:,1));
    ccI2 = complex_cells(I(:,:,2));
    ccI3 = complex_cells(I(:,:,3));
    ccI = ccI1 + ccI2 + ccI3;
    %figure(2);subplot(2,2,1);imshow(ccI);title("Complex cells");
    % Binarize
    thresC = 0.55;
    ccI(ccI > thresC) = 1;
    ccI(ccI ~= 1) = 0;
    % Edge 1
    edgeI1 = ccI;
    %figure(2);subplot(2,2,2);imshow(edgeI1);
    
    % 2-means clustering
    kmI = twomeanscluster(I);
    %figure(3);subplot(2,2,1);imshow(kmI);title("2-means");
    % Binarize
    clustercolor = kmI(1,1,:);
    tempI = zeros(maxR,maxC);
    for i = 1:maxR
        for j = 1:maxC
            if sum(clustercolor == kmI(i,j,:)) == 3
                tempI(i,j) = 1;
            else
                tempI(i,j) = 0;
            end
        end
    end
    kmI = tempI;
    %figure(3);subplot(2,2,2);imshow(kmI);
    % Clean
    kmI = bwareaopen(~kmI,120);
    kmI = bwareaopen(~kmI,120);
    kmI = imopen(kmI,strel("line",2,45));
    kmI = imclose(kmI,strel("line",2,45));
    %figure(3);subplot(2,2,3);imshow(kmI);
    % Edge 2
    edgeI2 = bwperim(kmI,8);
    edgeI2 = bwmorph(edgeI2,"bridge");
    %figure(3);subplot(2,2,4);imshow(edgeI2);
    
    % Entropy texture filter
    etI = entropyfilt(I);
    etI = etI./max(etI,[],"all");
    %figure(4);subplot(2,2,1);imshow(etI);title("Entropy filter");
    etI = im2gray(etI);
    %figure(4);subplot(2,2,2);imshow(etI)
    % Edge 3
    edgeI3 = edge(etI,"Canny",[0.25,0.30]);
    edgeI3 = bwmorph(edgeI3,"bridge");
    %figure(4);subplot(2,2,3);imshow(edgeI3);
    
    % Region-growing
    grown = zeros(maxR,maxC);
    rgI = zeros(maxR,maxC);
    thresR = 0.20;
    currLabel = 1;
    numRegion = 5;
    while sum(grown,"all") ~= maxR*maxC && numRegion > currLabel
        % choice seed
        [seedrow,seedcol] = find(~grown);
        seedrow = seedrow(mod(maxR,length(seedrow)+100));
        seedcol = seedcol(mod(maxC,length(seedrow)+100));
        currRegion = regiongrowing(im2gray(I),seedrow,seedcol,thresR);
        rgI = rgI + double(currRegion) * currLabel;
        grown = or(grown,currRegion);
        currLabel = currLabel + 1;
    end
    rgI = rgI./max(rgI,[],"all");
    %figure(5);subplot(2,2,1);imshow(rgI);title("Region growing");
    % Clean
    rgI = bwareaopen(~rgI,20);
    rgI = bwareaopen(~rgI,20);
    %figure(5);subplot(2,2,3);imshow(rgI);
    % Edge 4
    edgeI4 = edge(rgI,"Canny",0.25);
    edgeI4 = bwmorph(edgeI4,"bridge");
    %figure(5);subplot(2,2,4);imshow(edgeI4);
    
    % Refering
    rfI1 = refer(edgeI1,cat(3,edgeI2,edgeI3,edgeI4),2,0.065);
    %figure(7);subplot(2,2,1);imshow(rfI1);title("rfI1")
    rfI2 = refer(edgeI2,cat(3,edgeI1,edgeI3,edgeI4),5,0.065);
    %figure(7);subplot(2,2,2);imshow(rfI2);title("rfI2")
    rfI3 = refer(edgeI3,cat(3,edgeI1,edgeI2,edgeI4),5,0.035);
    %figure(7);subplot(2,2,3);imshow(rfI3);title("rfI3")
    rfI4 = refer(edgeI4,cat(3,edgeI1,edgeI2,edgeI3),5,0.035);
    %figure(7);subplot(2,2,4);imshow(rfI4);title("rfI4")
    % output
    seg = rfI1 + rfI2 + rfI3 + rfI4;
    seg(seg > 0) = 1;
    seg = logical(seg);
end

function [output] = refer(I,evidence,w,thres)
    % Let a set of edges found from one method refer to those 
    % found in other methods, each pixel is judged by projecting 
    % a gaussian area (mask) with size w to the evidence, and 
    % set to 1 if not below the threshold, 0 otherwise.
    mask = fspecial("gaussian",w,w/6);
    masked = zeros(size(I));
    for i = 1:size(evidence,3)
        masked(:,:,i) = conv2(evidence(:,:,i),mask,"same");
    end
    votes = sum(masked,3);
    votes(votes < thres) = 0;
    votes(votes >= thres) = 1;
    output = and(I,votes);
end

function [output] = twomeanscluster(I)
    % Proceed 2-means clustering to a RGB colour image, combining 
    % spacial information as an addtional feature.
    I = im2single(I);
    [coorxs,coorys] = meshgrid(1:size(I,2),1:size(I,1));
    features = cat(3,I,rgb2hsv(I),rgb2lab(I),coorxs,coorys);
    L = imsegkmeans(features,2,'NormalizeInput',true);
    output = label2rgb(L);
end

function [output] = complex_cells(I)
    % Process the image through complex cells at different orientation.
    ratio = min(size(I))/max(size(I));
    output = 0;
    for orien = 0:10:180
        ccell = complex_cell(I,1,0.75,orien,ratio,0);
        output = max(output,ccell);
    end
end

function [output] = complex_cell(I,sigma,freq,orient,aspect,phase)
    % Give output from a complex cell with given parameters to the 
    % quadrature pairs.
    [mask1,mask2] = quadrature_pairs(sigma,freq,orient,aspect,phase);
    convolved1 = conv2(I,mask1,'same');
    convolved2 = conv2(I,mask2,'same');
    output = sqrt(convolved1.^2 + convolved2.^2);
end

function [mask1,mask2] = quadrature_pairs(sigma,freq,orient,aspect,phase)
    % Generates a quadarture pairs with given parameters.
    % Acknowledgement:
    %   gabor2 function used is in a separated file and is taken 
    %   from cw4, which is not written by me.  
    % Output:
    %   mask1: a gabor with given parameters.
    %   mask2: a gabor with the given parameters, 
    %          with phase shifted by 90 degrees. 
    mask1 = gabor2(sigma,freq,orient,aspect,phase);
    mask2 = gabor2(sigma,freq,orient,aspect,phase+90);
end