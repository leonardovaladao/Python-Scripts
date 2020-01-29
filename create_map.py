# Map made with help of https://towardsdatascience.com/a-complete-guide-to-an-interactive-geographical-map-using-python-f4c5197e23e0

# Import dataset
from make_df import movies

# Create shapefile for map
import geopandas as gpd
shapefile = 'data/countries_110m/ne_110m_admin_0_countries.shp'
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.columns = ['country', 'country_code', 'geometry']

#Drop row corresponding to 'Antarctica'
gdf = gdf.drop(gdf.index[159])

# Merge shapefile object and movie dataframe
merged = gdf.merge(movies, left_on='country_code', right_on='Code', how='left')
merged.fillna('No data', inplace=True)

# Read data to json
import json
merged_json = json.loads(merged.to_json())

# Convert to string like object
json_data = json.dumps(merged_json)

# Import libraries for visualization
import io
from bokeh.plotting import figure
from bokeh.models import HoverTool, LinearColorMapper, ColorBar, GeoJSONDataSource
from bokeh.embed import components
from bokeh.resources import CDN
from jinja2 import Template

# Input GeoJSON source that contains features for plotting
geosource = GeoJSONDataSource(geojson=json_data)

#Define palette
palette = ['#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d']

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors. Input nan_color.
color_mapper = LinearColorMapper(palette = palette, nan_color = '#eeeeee',
                                 low = 5, high = 10)

#Add hover tool with HTML/CSS
hover = HoverTool(tooltips = """
                  <div class='main-box'>
                      <div class='head'>
                          <p style='font-family:Verdana; font-size:16px; font-weight:900;'> 
                              @movie_title
                          </p>
                          </br>
                          <img src='@Poster' style='width:150px; display:block; margin-left:auto; margin-right:auto;' alt=" "/>
                          </br>
                      </div>
                    <div class='information'>
                        <span style='font-weight:900;'>Country:</span> @country</br>
                        <span style='font-weight:900;'>IMDB Score:</span> @imdb_score</br>
                        <span style='font-weight:900;'>Language:</span> @Language</br>
                        <span style='font-weight:900;'>Year:</span> @Year</br>
                        <span style='font-weight:900;'>Genre:</span> @Genre</br>
                        <span style='font-weight:900;'>Director:</span> @Director</br>
                    </div>
                  </div>                  
                 """)

# Add template for HTML page
template = Template(
    '''<!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <title>A Country, A Movie</title>
                {{ resources }}
                {{ script }}
                <style>
                    .embed-wrapper {
                        display: flex;
                        justify-content: space-evenly;
                    }
                    h1{
                        top:30px;
                        bottom:0px;
                        text-align:center;
                        font-family: Verdana;
                    }
                </style>
            </head>
            <body>
                <h1>
                    Movies across the World, Ranked by IMDB Score
                </h1>
                <div class="embed-wrapper">
                    {{ div }}
                </div>
            </body>
        </html>
        ''')

#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
                     border_line_color=None,location = (0,0), orientation = 'horizontal')

#Create figure object.
p = figure( 
        plot_height = 650 , plot_width = 1050, 
        tools = [hover, 'pan', 'wheel_zoom', 'reset']
)

# Delete grids and axis
p.axis.visible = False
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource,fill_color = {'field' :'imdb_score', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 0.75)

# Add colorbar
p.add_layout(color_bar, 'below')

# Get bokeh parts
script_bokeh, div_bokeh = components(p)
resources_bokeh = CDN.render()

# Render everything together
html = template.render(resources=resources_bokeh,
                       script=script_bokeh,
                       div=div_bokeh)

# Save to HTML file
out_file_path = "map.html"
with io.open(out_file_path, mode='w') as f:
    f.write(html)