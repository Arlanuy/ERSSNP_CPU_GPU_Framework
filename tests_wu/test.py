#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit

from gpu_struct import GPUStruct
from mako.template import Template
from pycuda.compiler import SourceModule

src_template = Template(
"""
    struct Dist {
        %for s in xrange(ns):
        float *dist${s};
        %endfor
    };

    // return linear index based on x,y coordinate
    __device__ int get_index(int xcoord, int ycoord)
    {
        return ycoord + xcoord * ${ny};
    };

    __global__ void initialize(float *rho, float *ux, float *uy, Dist *ftmp)
    {
        int idx;
        float dens, velx, vely, vv, ev;

        for (int y = threadIdx.x + blockIdx.x * blockDim.x; 
                 y < ${ny}; 
                 y += blockDim.x * gridDim.x)   
        {   
            for (int x = threadIdx.y + blockIdx.y * blockDim.y; 
                 x < ${nx}; 
                 x += blockDim.y * gridDim.y) 
            {
                if ((x > 0) && (x < ${nx-1}) && (y > 0) && (y < ${ny-1}))
                {
                    idx = get_index(x,y);
                    dens = rho[idx]; velx = ux[idx]; vely = uy[idx];
                    vv = velx*velx + vely*vely;

                    %for s in xrange(ns):
                    // s = ${s}; \vec{e}[${s}] = [${ex[s]},${ey[s]}]
                    float *ftmp${s}_ptr1 = ftmp->dist${s};
                    ev = ${float(ex[s])}f*velx + ${float(ey[s])}f*vely;
                    ftmp${s}_ptr1[idx] = ${w[s]}f*dens*(1.0f+3.0f*ev+4.5f*ev*ev-1.5f*vv);                   
                    %endfor
                }
            }
        }
    }
"""
)

class channelFlow:
    # initialize channelFlow
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny

        max_threads_per_block = tools.DeviceData().max_threads
        self.blocksize = (ny if ny<32 else 32, nx if nx<32 else 32, 1)  # threads per block
        self.gridsize = (ny/self.blocksize[0], nx/self.blocksize[1], 1) # blocks per grid

        self.ns = 9
        self.w = np.array([4./9, 1./9, 1./9, 1./9, 1./9, 1./36, 1./36, 1./36, 1./36])
        self.ex = np.array([0, 1, -1, 0, 0, 1, -1, -1, 1])
        self.ey = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1])

        self.ctx = { 'nx': self.nx, 'ny': self.ny, 'ns': self.ns,
                     'w': self.w, 'ex': self.ex, 'ey': self.ey
                     }

        dtype = np.float32
        self.ftmp = np.zeros([self.nx,self.ny,self.ns]).astype(dtype)
        self.rho  = np.zeros([self.nx,self.ny]).astype(dtype)
        self.ux   = np.zeros([self.nx,self.ny]).astype(dtype)
        self.uy   = np.zeros([self.nx,self.ny]).astype(dtype)

        self.ftmp_gpu = GPUStruct([
            (np.float32,'*dist0', self.ftmp[:,:,0]),
            (np.float32,'*dist1', self.ftmp[:,:,1]),
            (np.float32,'*dist2', self.ftmp[:,:,2]),
            (np.float32,'*dist3', self.ftmp[:,:,3]),
            (np.float32,'*dist4', self.ftmp[:,:,4]),
            (np.float32,'*dist5', self.ftmp[:,:,5]),
            (np.float32,'*dist6', self.ftmp[:,:,6]),
            (np.float32,'*dist7', self.ftmp[:,:,7]),
            (np.float32,'*dist8', self.ftmp[:,:,8])
            ])
        self.rho_gpu  = cuda.mem_alloc(self.rho.nbytes)
        self.ux_gpu   = cuda.mem_alloc(self.ux.nbytes)
        self.uy_gpu   = cuda.mem_alloc(self.uy.nbytes)

    def run(self):
        src = src_template.render(**self.ctx)
        code = SourceModule(src)
        initialize  = code.get_function('initialize')

        self.rho[:,:] = 1.
        self.ux[:,:] = 0.
        self.uy[:,:] = 0.

        self.ftmp_gpu.copy_to_gpu()
        cuda.memcpy_htod(self.rho_gpu, self.rho)
        cuda.memcpy_htod(self.ux_gpu, self.ux)
        cuda.memcpy_htod(self.uy_gpu, self.uy)

        initialize(
            self.rho_gpu, self.ux_gpu, self.uy_gpu,
            self.ftmp_gpu.get_ptr(), 
            block=self.blocksize, grid=self.gridsize
            )

        self.dens = np.zeros_like(self.rho)
        cuda.memcpy_dtoh(self.dens, self.rho_gpu)
        print self.dens

if __name__ == "__main__":
    sim = channelFlow(1,2); sim.run()