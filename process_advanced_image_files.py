# as of Feb 2020
# digitalrocks.utexas.edu runs only python2, and does not have python3 nor numpy, csv, matplotlib, time installed

####
#
# TO-DO, meeting August 13, 2020
#
# 0. Masha: put this into github
#
# 1. (Bernie) Adapt routine below to read in .nc files - recognize is file has .nc extension, and read them in.
#(Automatically figure out that .nc is part of the file)
#
# 2. (Ningyu) Adapt routine below to read hdf5 file
#
# 3. (Masha) Adapt the routine below to create three orthonal cross-secttions
#
# 4. (separate, Alex) Assess which project cover images are too big (load slowly), create a list, 
#and then create a separate routine that reduces that file.
#
# 5. (separate, Anu/Bernie/Masha) Find projects with tiff slices, and then create a routine that in a given folder:
#    - recognizes how many consecutive tiff or jpeg slices are there, and then reads them and combines them into a volumetric tif.
#
# 6. Can we run all of these routines without showing images.
#
####


###############################################################################
### process_advanced_image_file.py
###
### Python3 script. Requires matplotlib, csv, time, sys, tifffile  modules.

### Authors: Hasan Khan and Masa Prodanovic, August 2018

### Developed with Python 3.6, numpy 1.14.3, tifffile 0.15.1, matplotlib 2.2.2
### Takes string input, resolves for the data types and imports accordingly
### It then produces histogram (as a plot and .csv file), thumbnail image and animated gif sequence. 
###  
### Works for raw (binary) files and volumetric tifs.
###
### Usage:
###          process_advanced_image_files.py
###            assumes default specific file (see below file)
###          process_advanced_image_files.py "in_file=FILENAME"
###            In this case the file is assumed to be volumetric tif file, 
###            and all of the required metadata is within the file.
###            The extension of the file has to be .tif or .tiff 
###
###          process_advanced_image_files.py "in_file=FILENAME&image_type=IMAGETYPE&width=NX&height=NY&number=NZ&byte_order=BYTEORDER" 
###            In this case, the file is assumed raw, packed 3D binary array.
###            FILENAME has not restrictions on the extension.
###            IMAGETYPE options: '8-bit Unisigned', '8-bit Signed',...
###            NX,NY,NZ are integers >=0 representing width, height and number of slices in z direction
###            BYTEORDER options: little-endian, big-endian
###
###           ../drp_upload_volume_test_data/ subdirectory outside of this repository
###           has some test datasets for these routines.
###
###############################################################################

import numpy as np
from matplotlib import pyplot as plt
import csv
import time
import matplotlib.animation as anim
import tifffile as tiff # module needs to be installed in Anaconda, from powershell type 'pip install tifffile'
import sys

print('Starting process_advanced_image_files.py')

t0 = time.time()

# Print arguments provided to the script
print('Number of arguments:', len(sys.argv),' arguments.')
print('Arguments provided:', str(sys.argv))

if len(sys.argv) == 1:
    # provide default input for testing since only name of the script was provided
    
    # raw image example
    #fileinput = "in_file=Ketton_rock_trapped_oil_Filtered_SSa.raw&image_type=16-bit Unsigned&width=630&height=410&number=510&byte_order=little-endian" 
    
    #volumetric tiff example
    fileinput = 'in_file=../drp_upload_volume_test_data/RLFeSO4_8bit_C2.tif'
    print('Assuming default input argument ',fileinput)
else:
    ## Process input argument.
    fileinput=sys.argv[1];

fileinput = fileinput.split("&")

filename, dtype, width, height, slices, byte_order, imagetype = 0, 0, 0, 0, 0, 0, 0

filename = fileinput[0].split("=")
filename = filename[1]

# Split filename and detect whether you find .tif or .tiff extension
basename = filename.split('.')
imagetype = basename[-1]

#print(len(fileinput))
   
## Read in as TIFF or RAW data file
# If we find .tif or .tiff extension, we assume volumetric tiff.
# In all other cases we assume raw binary file, which requires more metadata provided.
if ('tif' in imagetype or 'tiff' in imagetype):
    image = tiff.imread(filename)
    slices = image.shape[0]
    width = image.shape[1]
    height = image.shape[2]
    dt = image.dtype
    bt = image.dtype.byteorder
    alldata = image.reshape([np.size(image),])
else:
    dtype = fileinput[1].split("=")
    dtype = dtype[1]
    
    width = fileinput[2].split("=")
    width = int(width[1])
    
    height = fileinput[3].split("=")
    height = int(height[1])
    
    slices = fileinput[4].split("=")
    slices = int(slices[1])
    
    byte_order = fileinput[5].split("=")
    byte_order = byte_order[1]
    
        # Assign data type based on input
    if ('8' in dtype and 'Unsigned' in dtype): dt = 'u1'
    elif ('8' in dtype and 'unsigned' in dtype): dt = 'u1'
    
    elif ('16' in dtype and 'Unsigned' in dtype): dt = 'u2'
    elif ('16' in dtype and 'unsigned' in dtype): dt = 'u2'
    
    elif ('32' in dtype and 'Unsigned' in dtype): dt = 'u4'
    elif ('32' in dtype and 'unsigned' in dtype): dt = 'u4'
    
    elif ('8' in dtype and 'Signed' in dtype): dt = 'i1'
    elif ('8' in dtype and 'signed' in dtype): dt = 'i1'
    
    elif ('16' in dtype and 'Signed' in dtype): dt = 'i2'
    elif ('16' in dtype and 'signed' in dtype): dt = 'i2'
    
    elif ('32' in dtype and 'Signed' in dtype): dt = 'i4'
    elif ('32' in dtype and 'signed' in dtype): dt = 'i4'
    
    elif ('32' in dtype and 'Real' in dtype): dt = 'f4'
    elif ('32' in dtype and 'real' in dtype): dt = 'f4'
    
    elif ('64' in dtype and 'Real' in dtype): dt = 'f8'
    elif ('64' in dtype and 'real' in dtype): dt = 'f8'
    
    # Assign byte-order based on input
    if ('little' in byte_order or 'Little' in byte_order ): bt = '<'
    elif ('big' in byte_order or 'Big' in byte_order): bt = '>'
    else: bt = '|'
    
    datatype = bt + dt    
    alldata = np.fromfile(filename, dtype=datatype, sep="")
    image = alldata.reshape([slices, height, width])

t1 = time.time()
print('Time to read the file = '+str(t1-t0)+' sec')

# Generate histogram
#dataextent = np.max(image)-np.min(image)+1;
# So far, 256 bins works for all kinds of data. Adjust as needed.
nbins=256;
fig_hist = plt.figure(figsize=(4,2.4))
freq, bins, patches = plt.hist(alldata,nbins,density=True)
plt.xlabel('Gray value')
plt.ylabel('Probability')
plt.tight_layout()
plt.show()
fig_hist.savefig(filename+'.histogram.jpg',dpi=200)
print("Histogram .jpg generated")

with open(filename+'.csv','w',newline='') as csvfile:
	histwriter = csv.writer(csvfile,delimiter=',')
	histwriter.writerow(('Value','Probability'))
	for i in range(np.size(freq)):
		histwriter.writerow((bins[i],freq[i]))
print("Histogram .csv written")

t2 = time.time()
print('Histogram time = '+str(t2-t1)+' sec')

# scale data so that thumbnail and animated gif show correctly
# 8-bit range is typically required.
if dt is not 'uint8':
    #image min and max
    min_value = np.min(image);
    max_value = np.max(image);
    k=255/(max_value-min_value);
    l=-k*min_value;

    image1=np.floor(image*k + l)
    image = image1.astype('uint8')
    #print(image1.dtype)
    #print(image1)

    del image1 # remove the original image from memory
    
# Generate thumbnail
mydpi = 96
thumbslice = int(np.ceil(slices/2))
def make_image(data, size=(1, 1), dpi=96):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('Greys')
    ax.imshow(data, aspect='equal',vmin=0,vmax=255)
    
make_image(image[thumbslice,:,:],size=(width/mydpi,height/mydpi))
plt.savefig(filename+'.thumb.jpg', dpi=mydpi)
print("Thumbnail generated")

t3 = time.time()
print('Thumbnail time= '+str(t3-t2)+' sec')

# Draw and save GIF
class AnimatedGif:
	def __init__(self):
		self.fig = plt.figure()
		self.images = []

	def add(self, image,h,w,dpi=96):
		self.fig.set_size_inches(h/dpi,w/dpi)
		ax1 = plt.Axes(self.fig,[0., 0., 1., 1.])
		ax1.set_axis_off()
		self.fig.add_axes(ax1)
		plt.set_cmap('Greys')
		plt_im = ax1.imshow(image,vmin=0,vmax=255)
		self.images.append([plt_im])

	def save(self,filename):
		animation = anim.ArtistAnimation(self.fig,self.images)
		animation.save(filename,writer='imagemagick',fps=60)


sl1 = image[0,:,:]
images = []
animated_gif = AnimatedGif()
animated_gif.add(sl1, h=height, w=width)
print("Animation created for slice 1/"+str(slices))

if slices < 20:
    slicesave = 1
elif slices < 100:
    slicesave = 5
elif slices < 250:
    slicesave = 12
else:
    slicesave = 20

for i in range(1,slices,slicesave):
	sl = image[i,:,:]
	animated_gif.add(sl, h=height, w=width)
	print("Animation created for slice "+str(i+1)+"/"+str(slices))
animated_gif.save(filename+'.gif')

t4 = time.time()
print('Animation time = '+str(t4-t3)+' sec')
print('Total processing time = '+str(t4-t0)+' sec')
