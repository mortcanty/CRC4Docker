; docformat = 'rst'
; quality_index_run.pro
;    This program is free software; you can redistribute it and/or modify
;    it under the terms of the GNU General Public License as published by
;    the Free Software Foundation; either version 2 of the License, or
;    (at your option) any later version.
;
;    This program is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.

PRO quality_index_run_define_buttons, buttonInfo
   ENVI_DEFINE_MENU_BUTTON, buttonInfo, $
      VALUE = 'Quality Index', $
      REF_VALUE = 'CN Spectral Sharpening', $
      EVENT_PRO = 'quality_index_run', $
      UVALUE = 'QUALITY_INDEX',$
      POSITION = 'after', $
      /SEPARATOR
END

;+
; :Description:
;       ENVI extension for radiometric comparison of two
;       multispectral images
; :Params:
;      event:  in, required 
;         if called from the ENVI menu                 
; :Uses:
;       ENVI, PHASE_CORR, QUALITY_INDEX
; :Author:
;      Mort Canty (2013) 
;-
Pro quality_index_run, event

COMPILE_OPT IDL2
 

print, '------------------------------'
print, 'Quality Index'
print, systime(0)
print, '------------------------------'

catch, theError
if theError ne 0 then begin
   catch,/cancel
   ok = dialog_message(!Error_State.msg,/error)
   return
endif

envi_select, title='Choose reference image', fid=fid1, dims=dims1, pos=pos1
if (fid1 eq -1) then return
envi_file_query, fid1, fname=fname1, xstart=xstart1, ystart=ystart1, data_type=data_type1
map_info = envi_get_map_info(fid=fid1)
ps1 = map_info.ps[0]
print,'reference image: ',fname1

envi_select, title='Choose pan-sharpened image', fid=fid2, dims=dims2, pos=pos2
if (fid2 eq -1) then return
num_cols2 = dims2[2]-dims2[1]+1
num_rows2 = dims2[4]-dims2[3]+1
envi_file_query, fid2, fname=fname2, xstart=xstart2, ystart=ystart2, data_type=data_type2
map_info = envi_get_map_info(fid=fid2)
ps2 = map_info.ps[0]
print,'pan-sharpened image: ',fname2

if n_elements(pos1) ne n_elements(pos2) then begin
   print, 'Spectral subset sizes of the two images are different. Aborting.'
   m = dialog_message('Spectral subset sizes different',/error)
   return
endif

ratio = ps1/ps2

num_bands = n_elements(pos1)

; find upper left position of pan-sharpened image within reference image
envi_convert_file_coordinates,fid2,dims2[1],dims2[3],e,n,/to_map
envi_convert_file_coordinates,fid1,X_ul,Y_ul,e,n
X_ul = round(X_ul)
Y_ul = round(Y_ul)

; cutout the corresponding spatial subset of the reference image
num_cols1 = round(num_cols2/ratio)
num_rows1 = round(num_rows2/ratio)
dimsms = [-1L,X_ul,X_ul+num_cols1-1,Y_ul,Y_ul+num_rows1-1]
if (dimsms[1] lt dims1[1]) or (dimsms[3] lt dims1[3]) or $
   (dimsms[2] gt dims1[2]) or (dimsms[4] gt dims1[4]) then message,'MS dims out of bounds'

print, 'MS DIMS: ',dimsms[1:4]

im1 = fltarr(num_bands,num_cols1*num_rows1)
im2 = im1*0

txt = strarr(num_bands+1)
txt[0]= 'Wang-Bovik Quality Index
for k=0,num_bands-1 do begin
   b1 = envi_get_data(fid=fid1,dims=dimsms,pos=pos1[k])
; resample to reference image resolution
   b2 = envi_get_data(fid=fid2,dims=dims2,pos=pos2[k], $
                      interp=0, xfactor=1.0/ratio, yfactor=1.0/ratio)
   im2[k,*] = b2
; determine the offset (if any) by phase correlation
   offset = phase_corr(b1,b2,display=21)
; reselect (if necessary) the spatial subset of the reference image
   if (offset[0] ne 0) or (offset[1] ne 0) then begin
      dimsms[1:2] = dimsms[1:2] - offset[0]
      dimsms[3:4] = dimsms[3:4] - offset[1]
      if (dimsms[1] lt dims1[1]) or (dimsms[3] lt dims1[3]) or $
        (dimsms[2] gt dims1[2]) or (dimsms[4] gt dims1[4]) then message,'ms dims out of bounds'
      b1 = envi_get_data(fid=fid1,dims=dimsms,pos=pos1[k])
   endif
   im1[k,*] = b1
   txt[k+1] = 'band  '+strtrim(k+1,2)+ string(quality_index(b1,b2,blocksize=100))
endfor

envi_info_wid, txt, title='Fusion Quality'

end