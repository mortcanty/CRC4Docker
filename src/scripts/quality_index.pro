; docformat = 'rst'
; quality_index.pro
;    This program is free software; you can redistribute it and/or modify
;    it under the terms of the GNU General Public License as published by
;    the Free Software Foundation; either version 2 of the License, or
;    (at your option) any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details

function QI, x, y
   Sxy = correlate(x,y,/covariance)
   xm  = mean(x)
   ym  = mean(y)
   Sxx = variance(x)
   Syy = variance(y)
   return, 4*Sxy*xm*ym/((Sxx+Syy)*(xm^2+ym^2))
end

;+
; :Description:
;       Determine the Wang-Bovik quality index for a
;       pan-sharpened image band::
;         Ref: Wang and Bovik, IEEE Signal Processing
;         Letters9(3) 2002, 81-84
; :Params:
;      band1:  in, required 
;         reference spectral band
;      band2:  in, required   
;         degraded pan-sharpend spectral band               
; :Keywords:
;      blocksize: in, optional
;          size of image blocks to calculate 
;          index (default 8, i.e., 8 x 8)
; :Author:
;      Mort Canty (2009) 
;-
function quality_index, band1, band2, blocksize=blocksize
   if n_elements(blocksize) eq 0 then bs = 8 else bs = blocksize
   num_cols = (size(band1))[1] < (size(band2))[1]
   num_rows = (size(band1))[2] < (size(band2))[2]
   result = 0.0
   m = 0.0
   for j=0,num_cols-1-bs,bs/2 do for i=0,num_rows-1-bs,bs/2 do begin
      x = band1[j:j+bs-1,i:i+bs-1]
      y = band2[j:j+bs-1,i:i+bs-1]
      result = result + QI(x,y)
      m = m+1.0
   endfor
   return, result/m
end