import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymongo
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud, STOPWORDS

# Setting Webpage Configurations
st.set_page_config(page_icon="🌎",page_title="Airbnb", layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title(':rainbow[AirBnb Analysis] 🌍')

# READING THE CLEANED DATAFRAME

df = pd.read_csv(r"C:/Users/user/OneDrive/Desktop/Air _ BnB analysis/airbnb1.csv")

aggregated = df.groupby(['country','city']).count()

tab1,tab2,tab3,tab4= st.tabs(['Home','Geospatial Analysis 🌍','Exploratory Data Analysis for Price📈', 'Exploratory Data Analysis by Season📈'])

with tab1:
        
        col1,col2 = st.columns(2,gap= 'medium')
        col1.image("C:/Users/user/OneDrive/Desktop/airbnb_logo.png",width =300)
        col2.markdown("### :blue[Domain] : Travel Industry, Property Management and Tourism")
        col2.markdown("### :blue[Technologies used] : Python, Pandas, Plotly, Streamlit, MongoDB")
        col2.markdown("### :blue[Overview] : To analyze Airbnb data using MongoDB Atlas, perform data cleaning and preparation, develop interactive visualizations, and create dynamic plots to gain insights into pricing variations, availability patterns, and location-based trends. ")
        col2.markdown("#   ")
        col2.markdown("#   ")
        
with tab2:

        country_in = st.selectbox('Select a Country',options=df['country'].unique())

        if country_in == 'Australia':
            city_in = st.selectbox('Select a City',options=['Sydney'])

        elif country_in == 'Brazil':
            city_in = st.selectbox('Select a City',options=['Rio De Janeiro'])

        elif country_in == 'Canada':
            city_in = st.selectbox('Select a City',options=['Montreal'])

        elif country_in == 'China':
            city_in = st.selectbox('Select a City',options=['Hong Kong'])

        elif country_in == 'Hong Kong':
            city_in = st.selectbox('Select a City',options=['Hong Kong'])

        elif country_in == 'Portugal':
            city_in = st.selectbox('Select a City',options=['Porto','Other (International)'])

        elif country_in == 'Spain':
            city_in = st.selectbox('Select a City',options=['Barcelona'])

        elif country_in == 'Turkey':
            city_in = st.selectbox('Select a City',options=['Istanbul'])

        elif country_in == 'United States':
            city_in = st.selectbox('Select a City',options=['Kauai','Maui','New York','Oahu','The Big Island','Other (Domestic)'])

        sep1,sep2 = st.columns(2)

        with sep1:
            min_price = st.text_input('Select a Minimum Price (Min value : 9)')
        
        with sep2:
            max_price = st.text_input('Select a Maximum Price (Max value : 11,681)')

        st.divider() 

        if country_in and city_in and min_price and max_price:
            
            try:
                query_df = df.query(f'country == "{country_in}" and city == "{city_in}" and price>={min_price} and price<={max_price}')
                reset_index = query_df.reset_index(drop = True)
            
                # Creating map using folium
                base_latitude = reset_index.loc[0,'latitude']
                base_longitude = reset_index.loc[0,'longitude']

                base_map = folium.Map(location=[base_latitude,base_longitude], zoom_start=12)

                for index, row in reset_index.iterrows():
                    lat,lon = row['latitude'],row['longitude']
                    id = row['_id']
                    name = row['name']
                    price = row['price']
                    review = row['review_scores']
                    popup_text = f"ID: {id} | Name: {name} | Price: ${price} | Rating: {review}/10"
                    folium.Marker(location=[lat, lon], popup=popup_text).add_to(base_map)

                # call to render Folium map in Streamlit
                st_data = st_folium(base_map, width=1200,height = 600)

                st.divider()

                st.subheader(':orange[Top Hotels Recommended by Price and Ratings] 💸')

                query_top = df.query(f'country == "{country_in}" and city == "{city_in}"')

                query_1 = query_top.sort_values(by = ['price','review_scores'],ascending = False)

                new_df1 = query_1[['listing_url','name','city','country','amenities','price','review_scores','number_of_reviews']]

                st.dataframe(new_df1.head(10),hide_index=True,width = 1175, height = 220)

                st.divider()

                st.subheader(':orange[Enter the hotel name to know more about Availability] 📶')

                id_input = st.text_input('Enter a hotel name')

                if id_input:
                    new_query = reset_index.query(f'country == "{country_in}" and city == "{city_in}" and price>={min_price} and price<={max_price} and id == {id_input}')
                    new_table = new_query['listing_url','name','amenities','price','availability_30','availability_60','availability_90','availability_365','review_score','no_of_reviews']
                    st.dataframe(new_table,hide_index = True, width = 1175, height = 78)

                    st.divider()

            except:
                st.info('No results found')
                        
# Exploratory Data Analysis

with tab3:

    col1,col2 = st.columns(2)

    option = st.selectbox('Exploratory Data Analysis - Price Analysis',('Select an Analysis','Price Analysis by Country', 'Distribution of Price by Country', 'Distribution of Price using Box Plot','Scatter Plot by Price and Availability','Avg Price in Room_type BarChart'))
    unique_property_types = sorted(df['property_type'].unique())
    #selected_property_type = st.selectbox('Select Property_type', unique_property_types, index=0)
    #prop = st.selectbox('Select Property_type',sorted(df['property_type'].unique()),sorted(df['property_type'].unique()))

    st.divider() 
    # 1. Categorical Data -- country, city, review_score

    if option == 'Price Analysis by Country':
        fig = px.histogram(df,x = 'city',animation_frame='country',color = 'country')
        fig.update_layout(width=1200,height=500, title="Animated Histogram by City",xaxis_title="City",yaxis_title="Count")
        st.plotly_chart(fig)

        col1,col2 = st.columns(2)

        with col1:
        
            country_df = df[['country','city']].value_counts()
            new_country_df = pd.DataFrame(country_df,columns = ['Number of Hotels'])
            st.dataframe(new_country_df,width=450,height=528)

        with col2:
       
            grouped = df.groupby(['country','city']).agg({'price':'mean','review_scores':'mean'}).sort_values(by=['price','review_scores'],ascending = False)
            grouped = grouped.round()
            st.dataframe(grouped,width = 600,height = 528)
        st.divider() 

    # 2. Numerical Data - Price

    elif option == 'Distribution of Price by Country':

        # Distribution of Data in a Numerical Columns

        country = st.selectbox('Select any Country',options=df['country'].unique())

        if country == 'Australia':
            city = st.selectbox('Select any City',options=['Sydney'])

        elif country == 'Brazil':
            city = st.selectbox('Select any City',options=['Rio De Janeiro'])

        elif country == 'Canada':
            city = st.selectbox('Select any City',options=['Montreal'])

        elif country == 'China':
            city = st.selectbox('Select any City',options=['Hong Kong'])

        elif country == 'Hong Kong':
            city = st.selectbox('Select any City',options=['Hong Kong'])

        elif country == 'Portugal':
            city = st.selectbox('Select any City',options=['Porto'])

        elif country == 'Spain':
            city = st.selectbox('Select any City',options=['Barcelona'])

        elif country == 'Turkey':
            city = st.selectbox('Select any City',options=['Istanbul'])

        elif country == 'United States':
            city = st.selectbox('Select any City',options=['Kauai','Maui','New York','Oahu','The Big Island'])

        st.divider() 

        country_price_wise = df.query(f'country == "{country}" and city == "{city}"')

        plt.figure(figsize=(8,3.5))
        fig1 = sns.distplot(country_price_wise['price'])
        st.pyplot()

        if country_price_wise["price"].skew() > 0:
            st.write(f'Since the value is Positive : {country_price_wise["price"].skew()}, the curve is skewed Positively to the right side')
        elif country_price_wise["price"].skew() < 0:
            st.write(f'Since the value is Negative : {country_price_wise["price"].skew()}, the curve is skewed Negatively to the left side')
        
        st.divider() 

    if option == 'Distribution of Price using Box Plot':
        # Box_plot
        fig2 = px.box(df,x = 'country', y = 'price',color = 'country',width=1200, height=650)
        st.plotly_chart(fig2)

        st.divider() 

    elif option == 'Scatter Plot by Price and Availability':
        # 3. Scatter Plot (Numerical - Numerical) - (Price - Availability)

        subplots = make_subplots(rows=4, cols=1,subplot_titles = ('Availability 30', 'Availability 60', 'Availability 90', 'Availability 365'))
        scatter_plots1 = go.Scatter(x = df['price'],y = df['availability_30'],mode='markers', name = 'Availability 30')
        subplots.add_trace(scatter_plots1, row=1, col=1)
        
        scatter_plots2 = go.Scatter(x = df['price'],y = df['availability_60'],mode='markers',name = 'Availability 60')
        subplots.add_trace(scatter_plots2, row=2, col=1)

        scatter_plots3 = go.Scatter(x = df['price'],y = df['availability_90'],mode='markers',name = 'Availability 90')
        subplots.add_trace(scatter_plots3, row=3, col=1)

        scatter_plots4 = go.Scatter(x = df['price'],y = df['availability_365'],mode='markers',name = 'Availability 365')
        subplots.add_trace(scatter_plots4, row=4, col=1)

        subplots.update_layout(height=750,width = 1200) 
        st.plotly_chart(subplots)

        st.divider() 


with tab4:

    col1,col2 = st.columns(2)

    option = st.selectbox('Exploratory Data Analysis - By Seasons',('Select an Analysis','Bar Charts','Heatmaps','Bar Plot for Neighbourhood','Word Cloud'))

    st.divider() 

    if option == 'Bar Charts':
        # GETTING USER INPUTS
        country = st.sidebar.multiselect('Select a Country',sorted(df.country.unique()),sorted(df.country.unique()))
        prop = st.sidebar.multiselect('Select Property_type',sorted(df.property_type.unique()),sorted(df.property_type.unique()))
        room = st.sidebar.multiselect('Select Room_type',sorted(df.room_type.unique()),sorted(df.room_type.unique()))
        price = st.slider('Select Price',df.price.min(),df.price.max(),(df.price.min(),df.price.max()))
        
        # CONVERTING THE USER INPUT INTO QUERY
        query = f'country in {country} & room_type in {room} & property_type in {prop} & price >= {price[0]} & price <= {price[1]}'
        
        # CREATING COLUMNS
        col1,col2= st.columns(2,gap='medium')
        
        with col1:
            
        # TOP 10 PROPERTY TYPES BAR CHART

            df1 = df.query(query).groupby(["property_type"]).size().reset_index(name="listings").sort_values(by='listings',ascending=False)[:10]
            fig = px.bar(df1,
                            title='Top 10 Property Types',
                            x='listings',
                            y='property_type',
                            orientation='h',
                            color='property_type',
                            color_continuous_scale=px.colors.sequential.Agsunset)
            st.plotly_chart(fig,length=500,width=1000) 


            # TOP 10 HOSTS BAR CHART

            df2 = df.query(query).groupby(["host_name"]).size().reset_index(name="listings").sort_values(by='listings',ascending=False)[:10]
            fig = px.bar(df2,
                         title='Top 10 Hosts with Highest number of Listings',
                         x='listings',
                         y='host_name',
                         orientation='h',
                         color='host_name',
                         color_continuous_scale=px.colors.sequential.Agsunset)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig,use_container_width=True)


        with col2:
            
            # TOTAL LISTINGS IN EACH ROOM TYPES PIE CHART
            
            df1 = df.query(query).groupby(["room_type"]).size().reset_index(name="counts")
            fig = px.pie(df1,
                         title='total Listings in each Room_types',
                         names='room_type',
                         values='counts',
                         color_discrete_sequence=px.colors.sequential.Rainbow
                        )
            fig.update_traces(textposition='outside', textinfo='value+label')
            st.plotly_chart(fig,use_container_width=True)
            
            # Display the heatmap
    
    if option == 'Heatmaps':
            
            plt.figure(figsize=(10, 10))
            heatmap = sns.heatmap(df.groupby(['property_type', 'room_type']).price.mean().unstack(), annot=True, fmt=".0f", cmap=sns.cm.rocket_r, cbar_kws={'label': 'mean_price'})
            plt.xlabel('Room Type')
            plt.ylabel('Property Type')
            plt.title('Mean Price by Property Type and Room Type')
            st.pyplot()



            plt.figure(figsize=(12,12))
            heatmap1 = sns.heatmap(df.groupby(['property_type', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f", cmap = sns.cm.rocket_r, cbar_kws={'label': 'mean_price'})
            plt.ylabel('Property Type')
            plt.xlabel('bedrooms')
            plt.title('Mean Prices with Number of Bedrooms')
            st.pyplot()


    if option == 'Bar Plot for Neighbourhood':

        df_sorted = df.sort_values('host_total_listings_count', ascending=False)
        df_first_30 = df_sorted.head(50)

        objects = df_first_30['host_neighbourhood']
        y_pos = df_first_30['host_total_listings_count']

        df_first_30.plot(kind='bar', 
                        x='host_neighbourhood',
                        y='host_total_listings_count',
                        color='#66c2ff', 
                        figsize=(18, 8), 
                        title='Neighborhood Frequency', 
                        legend=False)

        plt.ylabel('Number_Of_Listings')
        st.pyplot()

    if option == 'Word Cloud':

        # Create a dataframe of the words that appear in the amenities section of the most expensive listings

        amenitiesDF = df[['amenities','price','_id',]]
        amenitiesDFTopper = amenitiesDF.sort_values('price',ascending=[0])
        amenitiesDFtop=amenitiesDFTopper.head(30)
        allemenities = ''
        for index,row in amenitiesDFtop.iterrows():
            p = re.sub('[^a-zA-Z]+',' ', row['amenities'])
            allemenities+=p

        allemenities_data=nltk.word_tokenize(allemenities)
        filtered_data=[word for word in allemenities_data if word not in stopwords.words('english')] 
        wnl = nltk.WordNetLemmatizer() 
        allemenities_data=[wnl.lemmatize(data) for data in filtered_data]
        allemenities_words=' '.join(allemenities_data)
       

        wordcloud = WordCloud(width = 1000, height = 700, background_color="white").generate(allemenities_words)
        plt.figure(figsize=(15,15))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot()


