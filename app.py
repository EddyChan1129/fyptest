from flask import Flask, request, render_template
import predict_stock as stock_analysis_module  # import your stock analysis functions

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        init_price = int(request.form['init_price'])
        results = stock_analysis_module.analyze_stock(ticker,init_price)
        return render_template('result.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

