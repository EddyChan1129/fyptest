from flask import Flask, request, render_template
import test as stock_analysis_module  # import your stock analysis functions

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        results = stock_analysis_module.analyze_stock(ticker)
        return render_template('result.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)