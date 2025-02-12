# cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from libc.stdio cimport FILE, fopen, fwrite, fclose, feof, fread
import os
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np

def export_mat_from_dict(counts, filename):
    cdef FILE* fout
    cdef int word1
    cdef int word2
    cdef double val
    cdef char* fn
    fn = filename
    fout = fopen(fn, 'w')
    for (i, c), v in counts.iteritems():
        word1 = i
        word2  = c
        val = v
        fwrite(&word1, sizeof(int), 1, fout) 
        fwrite(&word2, sizeof(int), 1, fout) 
        fwrite(&val, sizeof(double), 1, fout) 
    fclose(fout)

def export_mat_eff(np.ndarray[np.int32_t, ndim=1] row,
                  np.ndarray[np.int32_t, ndim=1] col,
                  np.ndarray[np.float64_t, ndim=1] data,
                  str filename):
    cdef FILE* fout
    cdef bytes fn = filename.encode('utf-8')  # Convert str to bytes
    cdef char* fn_ptr = fn
    cdef int i
    cdef int length = len(data)
    
    fout = fopen(fn_ptr, 'wb')  # Open in binary write mode
    if fout == NULL:
        raise IOError(f"Could not open file: {filename}")
    
    for i in range(length):
        fwrite(&row[i], sizeof(int), 1, fout)
        fwrite(&col[i], sizeof(int), 1, fout)
        fwrite(&data[i], sizeof(double), 1, fout)
    
    fclose(fout)


def retrieve_mat_as_coo(str matfn, min_size=None):
    """
    matfn = file name of matrix
    min_size = pad with zeros to this size
    """
    cdef FILE* fin
    cdef int word1, word2, ret
    cdef double val
    cdef bytes fn = matfn.encode('utf-8')
    cdef char* fn_ptr = fn
    
    fin = fopen(fn_ptr, 'rb')  # Binary mode
    if fin == NULL:
        raise IOError(f"Could not open file: {matfn}")
        
    cdef int size = (os.path.getsize(matfn) // 16)  # Integer division
    if min_size != None:
        size += 1
    cdef np.ndarray[np.int32_t, ndim=1] row = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(size, dtype=np.float64)
    cdef int i = 0
    
    while True:
        ret = fread(&word1, sizeof(int), 1, fin)
        if ret != 1:
            break
        ret = fread(&word2, sizeof(int), 1, fin)
        if ret != 1:
            break
        ret = fread(&val, sizeof(double), 1, fin)
        if ret != 1:
            break
            
        # Validate indices
        if word1 < 0 or word2 < 0:
            print(f"Warning: Invalid negative index at position {i}: ({word1}, {word2})")
            continue
            
        row[i] = word1
        col[i] = word2
        data[i] = val
        i += 1
        
        if i >= size:
            break
            
    fclose(fin)
    
    # Ensure we have valid data
    if i == 0:
        raise ValueError("No valid data read from file")
        
    # Trim arrays to actual size
    row = row[:i]
    col = col[:i]
    data = data[:i]
    
    # Add padding if needed
    if min_size is not None:
        if min_size > max(row.max(), col.max()) + 1:
            row = np.append(row, min_size - 1)
            col = np.append(col, min_size - 1)
            data = np.append(data, 0)
    
    # Final validation
    if row.min() < 0 or col.min() < 0:
        raise ValueError(f"Negative indices found: row_min={row.min()}, col_min={col.min()}")
    
    return coo_matrix((data, (row, col)), dtype=np.float64)
