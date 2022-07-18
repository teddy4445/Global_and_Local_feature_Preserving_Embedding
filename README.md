# Global and Local feature Preserving Embedding
A nonlinear dimension reduction method with both distance and neighborhood preservation.

## Implementation of the GLPE algorithm 
This repo implements the GLPE algorithm presented at:
```
Tan, C., Chen, C., Guan, J. (2013). A Nonlinear Dimension Reduction Method with Both Distance and Neighborhood Preservation. In: Wang, M. (eds) Knowledge Science, Engineering and Management. KSEM 2013. Lecture Notes in Computer Science, vol 8041. Springer, Berlin, Heidelberg.
```

## Usage example
```python
import pandas as pd
from glpe import GLPE
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("data_file_name.csv")
x = data.drop(["y_col_name"], axis=1)
y = data["y_col_name"]
reduced_x = GLPE.run(data=x, k=7, alpha=0.5)
model = KNeighborsRegressor(n_neighbors=7).fit(x, y)
```

## Dependencies
All the dependencies are stated in the requirements.txt file
1. Pandas
2. Numpy
3. Scipy
4. Scikit-learn
5. matplotlib - just for the example
6. seaborn - just for the example

## Install 
1. Download the repo
2. Install the requirements (pip install -r requirements.txt)
3. Run the project from the example.py (python example.py / python3 example.py) 