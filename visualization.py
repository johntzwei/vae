import os
import pickle

from bokeh.plotting import output_file, ColumnDataSource, figure, show
from bokeh.models import HoverTool, WheelZoomTool, PanTool, BoxZoomTool, ResetTool, SaveTool

EMBEDDINGS = 'runs/preliminary/embeddings.pkl'
save_dir = os.path.dirname(EMBEDDINGS)

embeddings = pickle.load(open(EMBEDDINGS, 'rb'))
source = ColumnDataSource(data=dict(
    x=[ i[1][0] for i in embeddings ],
    y=[ i[1][1] for i in embeddings ],
    S=[ ' '.join(i[0]) for i in embeddings ],
))

save_html = os.path.join(save_dir, 'visualization.html')
print('Saving visualization to %s' % save_html)
output_file(save_html)

hover = HoverTool(tooltips=[("S", "@S"),])
tools = [hover, WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool(), SaveTool()]
p = figure(tools=tools)
p.circle('x', 'y', size=10, source=source)
show(p)
