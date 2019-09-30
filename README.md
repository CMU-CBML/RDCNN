# RDCNN
RDCNN predicts the result of reaction-diffusion system based on convolutional neural network (CNN)

## User guide

User can run the code following the steps below.
          
### 1. FEM_Data_Geneartion (C++)

* **Description:** this code is used to generate the dataset using FEM.
* **Input:**
    * controlmesh.vtk (control mesh file)
* **Output:**
    * geometry_X_Y_input.txt (the input data storing boundary condition corresponding to Xth geometry and Yth parameter settings)
    * mesh_X.txt (the output data storing concentration results)
    * dataset_DKTGeo.txt (the library stores the parameter setting of each sample)
    
* **To compile:** (requires *[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)*)

    ` >> make`

* **To run:**

    Create a folder (eg. "\data\") first and then create another 3 folders in this folder ("\data\input\", "\data\output\" and "\data\parametric\") to store input data, output data and visualization results separately.

   ` >> ./rdfem -t <val_t> -s <val_s> -g <val_g> -o <output_path> ` 

   `output_path` is the output path for data generation

   Example: 

   `>> ./rdfem -t 100 -s 500 -g 21 -o ../data/`

### 2. txt_hdf5.ipynb 

* **Description:**
    This code is used to transform data format from TXT to H5
* **Input:**
    * ./data (Dataset folder)
* **Output:**
    * X.h5 (Dataset stored in h5 file)
* **To run：**
    User can open the code in jupyter notebook follow the comments to run the code.
        
### 3. Main_data_rdcnn2larger.ipynb

* **Description:** this code is used to train CNN model and predict concentration results for the specific reaction diffusion system.
                
* **To run：**
    User can open the code in jupyter notebook follow the comments to run the code.
