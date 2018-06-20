from sklearn import tree
from sklearn.datasets import load_iris
import pydotplus
from IPython.display import Image
iris = load_iris()

dtr = tree.DecisionTreeRegressor(max_depth = 5)
dtr.fit(iris.data,iris.target)

dot_data = tree.export_graphviz(
        dtr,
        out_file = None,
        feature_names = iris.feature_names,
        filled = True,
        impurity = False,
        rounded = True
        )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[6].set_fillcolor("#FFF2DD")
               

Image(graph.create_png())
	