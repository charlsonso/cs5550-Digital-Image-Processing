import numpy as np
from scipy.stats import hmean, trim_mean

#arithmetic mean filter
def amf(image, kernel_dim):
    (col, row) = image.shape
    new_image = np.zeros((col, row))
    for i in range(col):
        for j in range(row):
            k_distance_col = int(kernel_dim[0]/2)
            k_distance_row = int(kernel_dim[1]/2)
            least_col = i - k_distance_col
            least_row = j - k_distance_row
            most_col = i + k_distance_col
            most_row = j + k_distance_row
            if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
                start_col = least_col
                start_row = least_row
                end_col = most_col
                end_row = most_row
                if (least_col < 0):
                    start_col = 0
                if (least_row < 0):
                    start_row = 0
                if most_col >= col:
                    end_col = col
                if most_row >= row:
                    most_row = row
                sum_kernel = np.sum(image[start_col : end_col, start_row : end_row])
                
                new_image[i,j] = int(sum_kernel/(kernel_dim[0] * kernel_dim[1]))
            else:
                #do entire kernel
                sum_kernel = np.sum(image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1])
                new_image[i, j] = int(sum_kernel/(kernel_dim[0] * kernel_dim[1]))
    
    return np.uint8(new_image)

def gmf(image, kernel_dim):
    (col, row) = image.shape
    new_image = np.zeros((col, row))
    for i in range(col):
        for j in range(row):
            k_distance_col = int(kernel_dim[0]/2)
            k_distance_row = int(kernel_dim[1]/2)
            least_col = i - k_distance_col
            least_row = j - k_distance_row
            most_col = i + k_distance_col
            most_row = j + k_distance_row
            if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
                start_col = least_col
                start_row = least_row
                end_col = most_col
                end_row = most_row
                if (least_col < 0):
                    start_col = 0
                if (least_row < 0):
                    start_row = 0
                if most_col >= col:
                    end_col = col
                if most_row >= row:
                    most_row = row
                sum_kernel = np.prod(image[start_col : end_col, start_row : end_row])
                
                new_image[i,j] = int(sum_kernel**(1/(kernel_dim[0] * kernel_dim[1])))
            else:
                #do entire kernel
                sum_kernel = np.prod(image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1])
                new_image[i, j] = int(sum_kernel**(1/(kernel_dim[0] * kernel_dim[1])))
    
    return np.uint8(new_image)

def hmf(image, kernel_dim):
   (col, row) = image.shape
   new_image = np.zeros((col, row))
   for i in range(col):
       for j in range(row):
           k_distance_col = int(kernel_dim[0]/2)
           k_distance_row = int(kernel_dim[1]/2)
           least_col = i - k_distance_col
           least_row = j - k_distance_row
           most_col = i + k_distance_col
           most_row = j + k_distance_row
           if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
               start_col = least_col
               start_row = least_row
               end_col = most_col
               end_row = most_row
               if (least_col < 0):
                   start_col = 0
               if (least_row < 0):
                   start_row = 0
               if most_col >= col:
                   end_col = col
               if most_row >= row:
                   most_row = row
               h_mean = hmean(image[start_col : end_col, start_row : end_row], axis=None)
               
               new_image[i,j] = int(h_mean)
           else:
               #do entire kernel
               h_mean = hmean(image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1], axis=None)
               new_image[i, j] = int(h_mean)
   
   return np.uint8(new_image)

def chmf(image, kernel_dim):
   (col, row) = image.shape
   new_image = np.zeros((col, row))
   for i in range(col):
       for j in range(row):
           val_kernel = None
           k_distance_col = int(kernel_dim[0]/2)
           k_distance_row = int(kernel_dim[1]/2)
           least_col = i - k_distance_col
           least_row = j - k_distance_row
           most_col = i + k_distance_col
           most_row = j + k_distance_row
           if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
               start_col = least_col
               start_row = least_row
               end_col = most_col
               end_row = most_row
               if (least_col < 0):
                   start_col = 0
               if (least_row < 0):
                   start_row = 0
               if most_col >= col:
                   end_col = col
               if most_row >= row:
                   most_row = row
               val_kernel = image[start_col : end_col, start_row : end_row]
           else:
               #do entire kernel
               val_kernel = image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1]
           val_kernel = val_kernel.astype(int)
           new_image[i,j] = int(np.sum(np.square(val_kernel))/np.sum(val_kernel))
   
   return np.uint8(new_image)

def maxf(image, kernel_dim):
   (col, row) = image.shape
   new_image = np.zeros((col, row))
   for i in range(col):
       for j in range(row):
           val_kernel = None
           k_distance_col = int(kernel_dim[0]/2)
           k_distance_row = int(kernel_dim[1]/2)
           least_col = i - k_distance_col
           least_row = j - k_distance_row
           most_col = i + k_distance_col
           most_row = j + k_distance_row
           if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
               start_col = least_col
               start_row = least_row
               end_col = most_col
               end_row = most_row
               if (least_col < 0):
                   start_col = 0
               if (least_row < 0):
                   start_row = 0
               if most_col >= col:
                   end_col = col
               if most_row >= row:
                   most_row = row
               val_kernel = image[start_col : end_col, start_row : end_row]
           else:
               #do entire kernel
               val_kernel = image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1]
           new_image[i,j] = val_kernel.max()   
   return np.uint8(new_image)

def minf(image, kernel_dim):
   (col, row) = image.shape
   new_image = np.zeros((col, row))
   for i in range(col):
       for j in range(row):
           val_kernel = None
           k_distance_col = int(kernel_dim[0]/2)
           k_distance_row = int(kernel_dim[1]/2)
           least_col = i - k_distance_col
           least_row = j - k_distance_row
           most_col = i + k_distance_col
           most_row = j + k_distance_row
           if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
               start_col = least_col
               start_row = least_row
               end_col = most_col
               end_row = most_row
               if (least_col < 0):
                   start_col = 0
               if (least_row < 0):
                   start_row = 0
               if most_col >= col:
                   end_col = col
               if most_row >= row:
                   most_row = row
               val_kernel = image[start_col : end_col, start_row : end_row]
           else:
               #do entire kernel
               val_kernel = image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1]
           new_image[i,j] = val_kernel.min()
   return np.uint8(new_image)

def midf(image, kernel_dim):
   (col, row) = image.shape
   new_image = np.zeros((col, row))
   for i in range(col):
       for j in range(row):
           val_kernel = None
           k_distance_col = int(kernel_dim[0]/2)
           k_distance_row = int(kernel_dim[1]/2)
           least_col = i - k_distance_col
           least_row = j - k_distance_row
           most_col = i + k_distance_col
           most_row = j + k_distance_row
           if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
               start_col = least_col
               start_row = least_row
               end_col = most_col
               end_row = most_row
               if (least_col < 0):
                   start_col = 0
               if (least_row < 0):
                   start_row = 0
               if most_col >= col:
                   end_col = col
               if most_row >= row:
                   most_row = row
               val_kernel = image[start_col : end_col, start_row : end_row]
           else:
               #do entire kernel
               val_kernel = image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1]
           val_kernel = val_kernel.astype(int)
           min_val = val_kernel.min()
           max_val = val_kernel.max()
           new_image[i,j] = int((min_val + max_val) / 2)
   return np.uint8(new_image)

def atmf(image, kernel_dim):
   (col, row) = image.shape
   new_image = np.zeros((col, row))
   for i in range(col):
       for j in range(row):
           val_kernel = None
           k_distance_col = int(kernel_dim[0]/2)
           k_distance_row = int(kernel_dim[1]/2)
           least_col = i - k_distance_col
           least_row = j - k_distance_row
           most_col = i + k_distance_col
           most_row = j + k_distance_row
           if (least_col < 0 or least_row < 0 or most_col >= col or most_row >= row):
               start_col = least_col
               start_row = least_row
               end_col = most_col
               end_row = most_row
               if (least_col < 0):
                   start_col = 0
               if (least_row < 0):
                   start_row = 0
               if most_col >= col:
                   end_col = col
               if most_row >= row:
                   most_row = row
               val_kernel = image[start_col : end_col, start_row : end_row]
           else:
               #do entire kernel
               val_kernel = image[i - k_distance_col : i + k_distance_col + 1, j - k_distance_row : j + k_distance_row + 1]
           val_kernel = val_kernel.astype(int)
           val = int(trim_mean(val_kernel, 0.1, axis = None))
           print(val)
           new_image[i,j] = val
   return np.uint8(new_image)
