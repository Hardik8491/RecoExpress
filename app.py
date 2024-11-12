from flask import Flask, request, jsonify
from recommendation_model import RecommendationModel

app = Flask(__name__)
model = RecommendationModel()

@app.route('/recommend', methods=['GET'])
def recommend():
    product_name = request.args.get('product_name')
    num_recommendations = int(request.args.get('num_recommendations', 5))
    recommendations = model.recommend_products(product_name, num_recommendations)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
