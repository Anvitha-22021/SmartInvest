from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Load stock data
data = pd.read_csv("five_months_stock_data.csv")

stock_company_mapping = {
    'MMM': '3M Company',
    'ABT': 'Abbott Laboratories',
    'AA': 'Alcoa Corp',
    'ALL': 'The Allstate Corporation',
    'MO': 'Altria Group',
    'AXP': 'American Express Company',
    'AIG': 'American International Group Inc',
    'T': 'AT&T Inc',
    'BAC': 'Bank of America Corp',
    'BK': 'Bank of New York Mellon',
    'BAX': 'Baxter International Inc',
    'BRK-B': 'Berkshire Hathaway Inc',
    'BA': 'The Boeing Company',
    'BMY': 'Bristol-Myers Squibb Company',
    'CCL': 'Carnival Corporation',
    'CAT': 'Caterpillar Inc',
    'CVX': 'Chevron Corp',
    'C': 'Citigroup Inc'
}

stock_finviz_links = {
    'MMM': 'https://finviz.com/quote.ashx?t=MMM',
    'ABT': 'https://finviz.com/quote.ashx?t=ABT',
    'AA': 'https://finviz.com/quote.ashx?t=AA',
    'ALL': 'https://finviz.com/quote.ashx?t=ALL',
    'MO': 'https://finviz.com/quote.ashx?t=MO',
    'AXP': 'https://finviz.com/quote.ashx?t=AXP',
    'AIG': 'https://finviz.com/quote.ashx?t=AIG',
    'T': 'https://finviz.com/quote.ashx?t=T',
    'BAC': 'https://finviz.com/quote.ashx?t=BAC',
    'BK': 'https://finviz.com/quote.ashx?t=BK',
    'BAX': 'https://finviz.com/quote.ashx?t=BAX',
    'BRK-B': 'https://finviz.com/quote.ashx?t=BRK-B',
    'BA': 'https://finviz.com/quote.ashx?t=BA',
    'BMY': 'https://finviz.com/quote.ashx?t=BMY',
    'CCL': 'https://finviz.com/quote.ashx?t=CCL',
    'CAT': 'https://finviz.com/quote.ashx?t=CAT',
    'CVX': 'https://finviz.com/quote.ashx?t=CVX',
    'C': 'https://finviz.com/quote.ashx?t=C'
}

stock_image_links = {
    'MMM': 'https://images.livemint.com/img/2022/09/01/600x338/3M_1661994375391_1661994375616_1661994375616.jpg',
    'ABT': 'https://s3.amazonaws.com/medill.wordpress.offload/WP%20Media%20Folder%20-%20medill-reports-chicago/wp-content/uploads/sites/3/2020/06/ABBOTT-SIGN.jpg',
    'AA': 'https://upload.wikimedia.org/wikipedia/commons/b/bb/Alcoa_Corporation_Headquarters_-_Pittsburgh_%2848171783747%29.jpg',
    'ALL': 'https://finpedia.co/bin/download/Allstate%20Corporation%20/WebHome/ALL1.jpg?rev=1.1',
    'MO': 'https://static01.nyt.com/images/2019/08/27/science/27TOBACCO2/27TOBACCO2-jumbo.jpg?quality=75&auto=webp',
    'AXP': 'https://www.americanexpress.com/content/dam/amex/en-us/careers/images/Location-Home/NY1_Desktop.jpg',
    'AIG': 'https://finpedia.co/bin/download/American%20International%20Group%20%28AIG%29%20Inc/WebHome/AIG2.jpg?rev=1.1',
    'T': 'https://www.investopedia.com/thmb/KEOV74XQRCan9VMoKnWOVfyVdSA=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/INV_ATTStore_GettyImages-1544395513-282558b3889540f2ba5b46b5dec07578.jpg',
    'BAC': 'https://thumbor.forbes.com/thumbor/fit-in/1290x/https://www.forbes.com/advisor/wp-content/uploads/2022/09/getty_bank_of_america_near_me.jpeg.jpg',
    'BK': 'https://s.wsj.net/public/resources/images/BN-XG366_3n3Eh_MP_20180131161500.jpg',
    'BAX': 'https://images.ctfassets.net/7xz1x21beds9/7nZaSJaFs7N2exTifbg0FD/5c3e79de5b44f91f537ec629c3ac3d98/baxter-logo.jpg?w=870&h=470&q=90&fm=webp',
    'BRK-B': 'https://www.investopedia.com/thmb/F0hM4XCC8exzmd13oA_xMlgbprs=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GettyImages-940610188-64fbeeb4431e4932915a4b5daca7f1f4.jpg',
    'BA': 'https://img.etimg.com/thumb/msid-96151621,width-300,height-225,imgsize-87830,resizemode-75/file-photo-a-boeing-787-10-dreamliner-taxis-past-the-final-assembly-building-at-boeing-south-carolina-in-north-charleston.jpg',
    'BMY': 'https://www.pharmaceutical-technology.com/wp-content/uploads/sites/24/2023/07/bms-1.jpg',
    'CCL': 'https://nypost.com/wp-content/uploads/sites/2/2024/03/2012-carnival-parent-company-costa-32510034.jpg?resize=2048,1132&quality=75&strip=all',
    'CAT': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Caterpillarhq.JPG/440px-Caterpillarhq.JPG',
    'CVX': 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/i77edD_HBIgc/v1/2000x1333.jpg',
    'C': 'https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iWnm3MRGrbKk/v1/2000x1334.jpg'
}

# Load customer data
df = pd.read_csv('miniproject.csv')  
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(df[['Income', 'Savings']])
labels = kmeans.labels_

def predict_cluster(Income, Savings):
    customer_data = [[Income, Savings]]
    cluster = kmeans.predict(customer_data)
    return cluster[0]

# Get the list of unique tickers
tickers = data['ticker'].unique()

# Define the specific date
specific_date = "2024-04-24"

# Define a function to calculate potential returns and total amount in INR
def calculate_returns_and_total(initial_price, future_price, investment_amount):
    potential_return = (future_price - initial_price) / initial_price * 100
    total_amount = investment_amount * (1 + potential_return / 100)
    return round(potential_return, 2), total_amount

def categorize_returns(return_percentage):
    if return_percentage < 0:
        return 'Negative'
    elif return_percentage <= 4:
        return 'Low'
    elif return_percentage <= 15:
        return 'Medium'
    else:
        return 'High'

@app.route('/')
def index():
    return render_template('example.html')

@app.route('/newpage.html')
def newpage():
    return render_template('newpage.html')

@app.route('/index.html')
def index2():
    return render_template('example.html')

@app.route('/app.py', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting form data
        income = float(request.form['income'])
        savings = float(request.form['savings'])
        investment_amount = float(request.form['investment_amount'])
        investment_time = int(request.form['investment_duration'])
        
        # Validate the investment time input
        if not 1 <= investment_time <= 10:
            return render_template('error.html', message="Invalid investment time horizon. Please enter a value between 1 and 10.")
        
        # Define the number of days based on the investment time horizon
        days_in_investment_time = investment_time * 30

        # Initialize lists to store suggested stocks
        low_expected_return_stocks = []
        medium_expected_return_stocks = []
        high_expected_return_stocks = []

        # Iterate over each ticker
        for ticker in tickers:
            # Filter data for the current ticker
            ticker_data = data[data['ticker'] == ticker]

            # Extract features and target variable for training
            X_stock = ticker_data[['1. open', '2. high', '3. low', '5. volume', 'sentiment_score']]
            y_stock = ticker_data['4. close']

            # Get the price for the specific date if available
            specific_date_data = ticker_data[ticker_data['date'] == specific_date]
            if not specific_date_data.empty:
                specific_date_price = specific_date_data['4. close'].values[0]
            else:
                continue  # Skip to the next ticker if data is not available for the specific date

            # Train the Random Forest Regressor for stock price prediction
            model_stock = RandomForestRegressor(n_estimators=100, random_state=42)
            model_stock.fit(X_stock, y_stock)

            # Make predictions for the specified investment time horizon
            future_X = X_stock.tail(days_in_investment_time)
            future_y_pred = model_stock.predict(future_X)

            # Calculate potential returns and total amount
            potential_return, total_amount = calculate_returns_and_total(
                specific_date_price, future_y_pred[0], investment_amount)
            
            # Categorize expected returns
            return_category = categorize_returns(potential_return)

            # Cluster customers based on expected returns
            predicted_cluster = predict_cluster(income, savings)

            # Add the stock to the respective list based on the return category
            if return_category == 'Low':
                low_expected_return_stocks.append((ticker, potential_return, total_amount))
            elif return_category == 'Medium':
                medium_expected_return_stocks.append((ticker, potential_return, total_amount))
            elif return_category == 'High':
                high_expected_return_stocks.append((ticker, potential_return, total_amount))

        # Print the suggested stocks based on the predicted cluster
        if predicted_cluster == 1:
            suggestion = "Customer is in the low expected return category. Consider the following stocks:"
            low_expected_return_stocks.sort(key=lambda x: x[1], reverse=True)
            stocks = low_expected_return_stocks[:4]
        elif predicted_cluster == 2:
            suggestion = "Customer is in the medium expected return category. Consider the following stocks:"
            medium_expected_return_stocks.sort(key=lambda x: x[1], reverse=True)
            stocks = medium_expected_return_stocks[:4]
        else:
            suggestion = "Customer is in the high expected return category. Consider the following stocks:"
            high_expected_return_stocks.sort(key=lambda x: x[1], reverse=True)
            stocks = high_expected_return_stocks[:4]

        # Adjusted output to include investment duration
        duration_text = "month" if investment_time == 1 else "months"
        suggestion += f" (after {investment_time} {duration_text})"

        return render_template('result.html', suggestion=suggestion, stocks=stocks, stock_company_mapping=stock_company_mapping,stock_finviz_links=stock_finviz_links,stock_image_links =stock_image_links)
    else:
        # In case of a GET request to /app.py, redirecting to the index page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
