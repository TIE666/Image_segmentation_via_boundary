function gb=gabor2(sigma,freq,orient,aspect,phase)
%function gb=gabor2(sigma,freq,orient,aspect,phase)
%
% This function produces a numerical approximation to 2D Gabor function.
% Parameters:
% sigma  = standard deviation of Gaussian envelope, this in-turn controls the
%          size of the result (pixels)
% freq   = the frequency of the sin wave (1/pixels)
% orient = orientation of the Gabor from the vertical (degrees)
% aspect = aspect ratio of Gaussian envelope (1 = circular symmetric envelope,
%          lower values produce longer functions)
% phase  = the phase of the sin wave (degrees)

sz=fix(7*sigma./max(0.2,aspect));
if mod(sz,2)==0, sz=sz+1; end
 
[x y]=meshgrid(-fix(sz/2):fix(sz/2),fix(-sz/2):fix(sz/2));
 
% Rotation 
orient=orient*pi/180;
xDash=x*cos(orient)+y*sin(orient);
yDash=-x*sin(orient)+y*cos(orient);

phase=phase*pi/180;

gb=exp(-.5*((xDash.^2/sigma^2)+(aspect^2*yDash.^2/sigma^2))).*(cos(2*pi*xDash*freq+phase));

%ensure gabor sums to zero (so it is a valid differencing mask)
gbplus=max(0,gb);
gbneg=max(0,-gb);
gb(gb>0)=gb(gb>0)./sum(sum(gbplus));
gb(gb<0)=gb(gb<0)./sum(sum(gbneg));
