#compression ratio and encoding time
import time
import sys
import numpy as np
import heapq

def run_length_encoding_gray(image):
    t0 = time.time()
    
    size = image.shape
    current_val = image[0][0]
    count = 1
    encoded_str = ""
    encoded_str += str(current_val)
    for w in range(size[1]):
        for h in range(size[0]):
            if w == 0 and h == 0:
                continue
            pixel_val = image[h, w]
            if pixel_val == current_val:
                count+=1
            else:
                encoded_str += "v"+str(current_val)+"l"+str(count)
                current_val = pixel_val
                count = 1
    t1 = time.time()
    total_time = t1-t0

    image_size_bits = size[0]*size[1]*8
    compressed_bits = len(encoded_str)*16
    ratio = float(image_size_bits)/float(compressed_bits)
    print('done')
    return (encoded_str, ratio, total_time)

def run_length_encoding_bit_plane(image):
    t0 = time.time()
    encoded_str = ""
    bit_plane_str = []
    for i in range(8):
        bit_str = ""
        for w in range(image.shape[1]):
            for h in range(image.shape[0]):
                pixel_val = image[h,w]
                if pixel_val > 2**(7-i):
                    image[h,w] = image[h,w] - 2**i
                    bit_str += "1"
                else:
                    bit_str += "0"
        bit_plane_str.append(bit_str)
    
    for i in range(len(bit_plane_str)):
        bit_plane = bit_plane_str[i]
        current_val = bit_plane[0]
        count = 1
        for j in range(len(bit_plane)):
            encoded_str += "p" + str(j)
            if j == 0:
                continue
            if current_val == bit_plane[j]:
                count += 1
            else:
                encoded_str += "v" + str(current_val) + "l" + str(count)
                current_val = bit_plane[j]
                count = 1

    image_size_bits = image.shape[1]*image.shape[0]*8
    compressed_bits = len(encoded_str)*16
    
    ratio = float(image_size_bits)/float(compressed_bits)
    t1 = time.time()
    total_time = t1-t0
    return (encoded_str, ratio, total_time)

def huffman_encoding(image):
    t0 = time.time() 

    table = {}
    for i in range(256):
        table[i] = 0
    size = image.shape
    for w in range(image.shape[1]):
        for h in range(image.shape[0]):
            table[image[h, w]] += 1
    
    heap = []
    codes = {}
    reverse_mapping = {}
    make_heap(heap, table)
    merge_nodes(heap)
    make_codes(heap, codes, reverse_mapping)
    encoded_txt = get_encoded_text(image, codes)
    t1 = time.time()
    total_time = t1 - t0
    image_size_bits = image.shape[1]*image.shape[0]
    compressed_bits = len(encoded_txt)
    ratio = float(image_size_bits)/float(compressed_bits)
    print('done')
    return (encoded_txt, codes, ratio, total_time)

def make_heap(heap, table):
    for key in table:
        node = HeapNode(key, table[key])
        heapq.heappush(heap, node)

def merge_nodes(heap):
    while(len(heap)>1):
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)

        merged = HeapNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2

        heapq.heappush(heap, merged)

def make_codes(heap, codes, reverse_mapping):
    root = heapq.heappop(heap)
    current_code = ""
    make_codes_helper(root, current_code, codes, reverse_mapping)

def make_codes_helper(root, current_code, codes, reverse_mapping):
    if (root == None):
        return

    if (root.val != None):
        codes[root.val] = current_code
        reverse_mapping[current_code] = root.val
        return
    
    make_codes_helper(root.left, current_code + "0", codes, reverse_mapping)
    make_codes_helper(root.right, current_code + "1", codes, reverse_mapping)

def get_encoded_text(image, codes):
    encoded_text = ""
    for w in range(image.shape[1]):
        for h in range(image.shape[0]):
            encoded_text += str(codes[image[h,w]])
    return encoded_text

class HeapNode:
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, HeapNode)):
            return False
        return self.freq == other.freq
