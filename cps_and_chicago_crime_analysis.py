"""Capstone Personal Project: Save the Children, Save the City.

In this script, we will try to find an optimal location to place a new
nonprofit community center, HERO (Helping Everyone Receive Opportunities),
in Chicago. Specifically, this analysis will be intended for stakeholders
interested in opening a non-profit educational community center to combat the
high crime rates, provide aid to impoverished communities, and keep the youth
of the perilous areas of Chicago off the streets..

Script prints tables and exports graph .png files Machine Learning algorithms
and data visualization techniques from Folium and Matplotlib to analyze crime,
socioeconomic status, and CPS data in Chicago.

NOTE: GoeJSON file below needs to be downloaded in the same repository:
#!wget --quiet  https://raw.githubusercontent.com/RandomFractals/
ChicagoCrimes/master/data/chicago-community-areas.geojson -O chi_geojson.json
"""


import io
import folium
import requests
import sklearn.utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
from IPython.display import Image
from pandas.io.json import json_normalize
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import DBSCAN
from pylab import rcParams
from sklearn.preprocessing import StandardScaler
from secrets import client_id, client_secret


class CrimeCPSAnalysis:
    """Data analysis of Chicago Crime and CPS."""

    def __init__(self):
        """Initialize shared variables."""
        self.latitude, self.longitude = self.search_for_cps_locations()
        self.schools_df = self.clean_cps_data()
        self.merged_df = self.clean_chi_crime_data()
        self.clusters_df = self.cluster_data()

    def search_for_cps_locations(self):
        """Use Foursquare API to get locations of all CPS."""
        address = 'Chicago, IL'
        geolocator = Nominatim(user_agent="Chicago_explorer")
        location = geolocator.geocode(address)
        latitude = location.latitude
        longitude = location.longitude

        # search for schools using foursquare apo
        search_query = 'public school'
        radius = 500

        # Foursquare API
        url = 'https://api.foursquare.com/v2/venues/'\
              'search?client_id={}&client_secret={}&ll={},'\
              '{}&v={}&query={}&radius={}&limit={}'.format(
                  client_id,
                  client_secret,
                  latitude,
                  longitude,
                  '20180604',
                  search_query,
                  radius,
                  30)
        # Save results in json style
        results = requests.get(url).json()
        # assign relevant part of JSON to venues
        venues = results['response']['venues']
        # tranform venues into a dataframe
        venues_df = json_normalize(venues)
        print("CPS Foursquare data pull:\n", venues_df.head())

        return latitude, longitude

    def clean_cps_data(self):
        """Clean and wrangle Chicago Public Schools Data."""
        url1 = 'https://data.cityofchicago.org/resource/9xs2-f89t.csv'
        cps_df = pd.read_csv(url1)

        # Clean CPS df to only show the relevant data for our investigation
        schools_df = cps_df[['name_of_school', 'elementary_or_high_school',
                            'community_area_name', 'community_area_number',
                             'latitude', 'longitude', 'safety_score',
                             'environment_score', 'instruction_score',
                             'rate_of_misconducts_per_100_students_',
                             'graduation_rate_', 'college_eligibility_']]
        schools_df.rename(columns={'latitude': 'school_lat',
                                   'longitude': 'school_long'}, inplace=True)
        schools_df.columns = ['name of school', 'elementary or high school',
                              'community area name', 'community area number',
                              'school lat', 'school long', 'safety score',
                              'environment score', 'instruction score',
                              'rate of misconducts per 100 students',
                              'graduation rate', 'college eligibility']

        # replace str NDA value with None to convert dtype tot float
        schools_df.replace({'NDA': None}, inplace=True)
        schools_df['graduation rate'] = schools_df[
            'graduation rate'].astype('float64')
        schools_df['college eligibility'] = schools_df[
            'college eligibility'].astype('float64')

        # Group the df by community name & take the mean of the values
        schools_gdf = schools_df.groupby(
            'community area name', as_index=False).mean()

        # create map of Chicgo using school latitude and longitude values
        map_chicago_schools = folium.Map(location=[self.latitude,
                                                   self.longitude])

        # add markers to map
        for lat, lng, community, environment in zip(
            schools_gdf['school lat'],
            schools_gdf['school long'],
            schools_gdf['community area name'],
                schools_gdf['environment score']):
            label = '{} , score: {}'.format(community, environment)
            label = folium.Popup(label, parse_html=True)
            folium.CircleMarker(
                [lat, lng],
                radius=5,
                popup=label,
                color='blue',
                fill=True,
                fill_color='#3186cc',
                fill_opacity=0.7,
                parse_html=False).add_to(map_chicago_schools)

        # Export Map object as .png with 10 sec render time
        img_data = map_chicago_schools._to_png(10)
        img = Image.open(io.BytesIO(img_data))
        img.save('map_chicago_schools.png')

        return schools_df

    def clean_chi_crime_data(self):
        """Clean and wrangle Chicago Crime data."""
        # Now let's look at the crimes in each community area
        c_url = 'https://data.cityofchicago.org/resource/crimes.json'
        cc_df = pd.read_json(c_url)
        crimes_df = cc_df[['id', 'primary_type', 'community_area', 'latitude',
                           'longitude']]
        crimes_df.rename(columns={'id': 'crime_id',
                                  'community_area': 'community_area_number',
                                  'latitude': 'crime_lat',
                                  'longitude': 'crime_long'}, inplace=True)
        crimes_df.columns = ['crime id', 'primary type',
                             'community area number',
                             'crime lat', 'crime long']
        # Merge the schools and crime dataframes
        merge_df = pd.merge(self.schools_df, crimes_df,
                            on='community area number')

        # Combine the school and crime datasets containing releveant columns
        # Then take the mean of the scores/rates, count the number of crimes
        # per community, as well as combine the str data

        mrg_group_df = merge_df.groupby(
            ['community area number', 'community area name'],
            as_index=False, sort=False).agg(
            {'name of school': ','.join, 'school lat': 'mean',
             'school long': 'mean', 'primary type': ','.join,
             'crime id': 'count', 'crime lat': 'mean', 'crime long': 'mean',
             'safety score': 'mean', 'environment score': 'mean',
             'instruction score': 'mean',
             'rate of misconducts per 100 students': 'mean',
             'graduation rate': 'mean', 'college eligibility': 'mean'})
        mrg_group_df.rename(columns={'crime id': 'number of crimes'},
                            inplace=True)

        return mrg_group_df

    def cluster_data(self):
        """Clustering Data via DBSCAN."""
        # cleaning
        # remove rows that dont have any value in the number of crimes field
        df = self.merged_df[pd.notnull(
            self.merged_df['rate of misconducts per 100 students'])]
        df = df.reset_index(drop=True)
        df.rename(columns={'school long': 'Long', 'school lat': 'Lat',
                           'number of crimes': 'crimes'}, inplace=True)

        # Visualization of stations on map using basemap package.
        rcParams['figure.figsize'] = (14, 10)

        llon = -88.0817
        ulon = -87.0646
        llat = 41.525
        ulat = 42.460445

        df = df[(df['Long'] > llon) &
                (df['Long'] < ulon) &
                (df['Lat'] > llat) &
                (df['Lat'] < ulat)]

        my_map = Basemap(projection='merc',
                         resolution='l',
                         area_thresh=1000.0,
                         # min longitude (llcrnrlon) and latitude (llcrnrlat)
                         llcrnrlon=llon,
                         llcrnrlat=llat,
                         # max longitude (urcrnrlon) and latitude (urcrnrlat)
                         urcrnrlon=ulon,
                         urcrnrlat=ulat)

        my_map.drawcoastlines()
        my_map.drawcountries()
        my_map.drawstates()
        my_map.fillcontinents(color='white', alpha=0.3)
        my_map.shadedrelief()

        # To collect data based on stations
        xs, ys = my_map(np.asarray(df.Long), np.asarray(df.Lat))
        df['xm'] = xs.tolist()
        df['ym'] = ys.tolist()

        # Visualization1
        for index, row in df.iterrows():
            my_map.plot(row.xm,
                        row.ym,
                        markerfacecolor=([1, 0, 0]),
                        marker='o',
                        markersize=5,
                        alpha=0.75)
        # Save Figure
        plt.savefig('rate_of_misconduct_cluster_map.png')

        # Clustering of communities based on thier location, number of crimes,
        # environment score and graduation rate
        sklearn.utils.check_random_state(1000)
        clus_dataset = df[['xm', 'ym', 'rate of misconducts per 100 students']]
        clus_dataset = np.nan_to_num(clus_dataset)
        clus_dataset = StandardScaler().fit_transform(clus_dataset)

        # Compute DBSCAN
        db = DBSCAN(eps=0.7, min_samples=4).fit(clus_dataset)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        df["Clus_Db"] = labels

        clusternum = len(set(labels))

        # Add clusters to the basemap visualization
        rcParams['figure.figsize'] = (14, 10)
        my_map = Basemap(projection='merc',
                         resolution='l', area_thresh=1000.0,
                         llcrnrlon=llon, llcrnrlat=llat,
                         urcrnrlon=ulon, urcrnrlat=ulat)

        my_map.drawcoastlines()
        my_map.drawcountries()
        my_map.fillcontinents(color='white', alpha=0.3)
        my_map.shadedrelief()

        # To create a color map
        colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusternum))

        # Visualization2
        for clust_number in set(labels):
            c = (([0.4, 0.4, 0.4]) if clust_number == -1 else colors[np.int(
                clust_number)])
            clust_set = df[df.Clus_Db == clust_number]
            my_map.scatter(clust_set.xm,
                           clust_set.ym,
                           color=c,
                           marker='o',
                           s=20,
                           alpha=0.85)
            if clust_number != -1:
                cenx = np.mean(clust_set.xm)
                ceny = np.mean(clust_set.ym)
                plt.text(cenx,
                         ceny,
                         str(clust_number),
                         fontsize=25,
                         color='red',)

        # A sample of clusters
        df2 = df[['Lat', 'Long', 'safety score', 'environment score',
                  'instruction score', 'rate of misconducts per 100 students',
                  'graduation rate', 'college eligibility', 'xm', 'ym',
                  'Clus_Db']]
        df2.rename(columns={
            'rate of misconducts per 100 students': 'misconducts_rate'},
            inplace=True)
        df2.reset_index(inplace=True)

        df3 = df2.groupby(['Clus_Db'], as_index=False).mean()

        # drop the outliers
        df3.drop(index=0, inplace=True)
        df3 = df3.round(
            decimals={'safety score': 0, 'environment score': 0,
                      'instruction score': 0, 'misconducts_rate': 2,
                      'graduation rate': 2, 'college eligibility': 2})
        df3 = df3.reset_index(drop=True)

        return df3

    def cluster_bar_chart(self):
        """Create bar graph to show what clusters mean."""
        df = self.clusters_df
        width = 0
        x = np.arange(4)  # the label locations
        ax = df[['safety score', 'environment score', 'instruction score',
                 'misconducts_rate', 'graduation rate',
                 'college eligibility']].plot(
            kind='bar',
            figsize=(20, 8),
            width=0.3,
            color=['red', 'green', 'orange', 'blue', 'purple', 'pink'],
            fontsize=14)

        ax.bar(x, df['safety score'], width)
        ax.bar(x, df['environment score'], width)
        ax.bar(x, df['instruction score'], width)
        ax.bar(x, df['misconducts_rate'], width)
        ax.bar(x, df['graduation rate'], width)
        ax.bar(x, df['college eligibility'], width)

        ax.set_title('Average Scores and Rates of CPS per Cluster',
                     fontsize=16)
        ax.set_xlabel('DB Cluster')

        ax.legend(['safety score', 'environment score', 'instruction score',
                   'rate of misconducts per 100 students''graduation rate',
                   'college eligibility'],
                  loc='upper right', fontsize=14)

        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

    def cluster_on_folium(self):
        """Map out the cluster on folium map."""
        coordinates = self.clusters_df[['Lat', 'Long']].values
        colorslist = ['red', 'blue', 'green', 'orange']

        # create map of Chicgo using school latitude and longitude values
        map_chicago = folium.Map(location=[self.latitude, self.longitude],
                                 zoom_start=8)

        # Add markers to map
        for i, j in enumerate(coordinates):
            folium.Marker(
                location=[j[0], j[1]],
                popup=folium.Popup(
                    '{} misconducts per 100 student, '
                    '{} % that are college eligible'.format(
                        self.clusters_df['misconducts_rate'][i],
                        self.clusters_df['college eligibility'][i]),
                    parse_html=True),
                icon=folium.Icon(colorslist[i])
            ).add_to(map_chicago)

        # Export Map object as .png with 10 sec render time
        img_data = map_chicago._to_png(10)
        img = Image.open(io.BytesIO(img_data))
        img.save('map_chicago_schools_cluster.png')

        # Add choropleth of crimes per community area
        cdf = self.merged_df.groupby(
            'community area number', as_index=False)['crime id'].count()
        cdf['community area number'] = cdf['community area number'].astype(
            'str')

        # Map out the cluster on folium map
        # create map of Chicgo using school latitude and longitude values
        map_chicago2 = folium.Map(location=[self.latitude, self.longitude],
                                  zoom_start=10)

        crime_geo = r'chi_geojson.json'  # File location in description

        # create a numpy array of length 6 and has linear spacing from the
        # minium total immigration to the maximum total immigration
        threshold_scale = np.linspace(cdf['crime id'].min(),
                                      cdf['crime id'].max(),
                                      6, dtype=int)
        threshold_scale = threshold_scale.tolist()
        # make sure that the last value of the list is greater than the
        # maximum immigration
        threshold_scale[-1] = threshold_scale[-1] + 1

        map_chicago2.choropleth(
            geo_data=crime_geo,
            data=cdf,
            columns=['community area number', 'crime id'],
            key_on='feature.properties.area_num_1',
            fill_color='YlOrRd',
            legend_name='Crime rate in Chicago',
            fill_opacity=0.7,
            line_opacity=0.2,
            threshold_scale=threshold_scale,
        )

        for i, j in enumerate(coordinates):
            folium.Marker(
                location=[j[0], j[1]],
                popup=folium.Popup('{} misconducts per 100 student, '
                                   '{} % that are college eligible'.format(
                                       self.merged_df['misconducts_rate'][i],
                                       self.merged_df['college eligibility'][i]
                                   ),
                                   parse_html=True),
                icon=folium.Icon(colorslist[i])
            ).add_to(map_chicago2)

        # Export Map object as .png with 10 sec render time
        img_data = map_chicago2._to_png(10)
        img = Image.open(io.BytesIO(img_data))
        img.save('map_chicago_schools_cluster2.png')

    def socioeceonomic_choropleth(self):
        """School data centriods over socioeconomic choropleth."""
        # Now lets pull chicago's socioeconomic data
        s_url = 'https://data.cityofchicago.org/resource/kn9c-c2s2.json'
        soc_df = pd.read_json(s_url)
        soc_df = soc_df[['community_area_name', 'per_capita_income_']]
        soc_df['community_area_name'] = soc_df[
            'community_area_name'].str.upper()
        soc_df = soc_df.groupby(
            'community_area_name', as_index=False)['per_capita_income_'].mean()

        # Map out the cluster on folium map
        # create map of Chicgo using school latitude and longitude values
        coordinates = self.clusters_df[['Lat', 'Long']].values
        map_chicago_s = folium.Map(location=[self.latitude, self.longitude],
                                   zoom_start=10)

        crime_geo = r'chi_geojson.json'

        # create a numpy array of length 6 and has linear spacing from the
        # minium total immigration to the maximum total immigration
        threshold_scale = np.linspace(soc_df['per_capita_income_'].min(),
                                      soc_df['per_capita_income_'].max(),
                                      6, dtype=int)
        threshold_scale = threshold_scale.tolist()
        threshold_scale[-1] = threshold_scale[-1] + 1

        map_chicago_s.choropleth(
            geo_data=crime_geo,
            data=soc_df,
            columns=['community_area_name', 'per_capita_income_'],
            key_on='feature.properties.community',
            fill_color='YlOrRd',
            legend_name='Per Capita Income',
            fill_opacity=0.7,
            line_opacity=0.2,
            threshold_scale=threshold_scale,
        )

        colorslist = ['red', 'blue', 'green', 'orange']

        for i, j in enumerate(coordinates):
            folium.Marker(
                location=[j[0], j[1]],
                popup=folium.Popup('{} misconducts per 100 student, '
                                   '{} % that are college eligible'.format(
                                       self.clusters_df['misconducts_rate'][i],
                                       self.clusters_df[
                                           'college eligibility'][i]),
                                   parse_html=True),
                icon=folium.Icon(colorslist[i])
            ).add_to(map_chicago_s)

        # Export Map object as .png with 10 sec render time
        img_data = map_chicago_s._to_png(10)
        img = Image.open(io.BytesIO(img_data))
        img.save('map_chicago_socioeconomic.png')


if __name__ == '__main__':
    chi = CrimeCPSAnalysis()
    chi.cluster_data()
    chi.cluster_on_folium()
    chi.socioeceonomic_choropleth()
