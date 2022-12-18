def predict_profit(population, weight, bias):
    
    ### Profit = weight * Population + bias
    
    return weight*population + bias




def cost_function(population, profit, weight, bias):
    
    # 𝑀𝑆𝐸=1/𝑁(∑𝑖=1𝑛(𝑦𝑖−(𝑚𝑥𝑖+𝑏))^2)
    
    area = len(population)
    total_error = 0.0
    for i in range(area):
        total_error += (profit[i] - (weight*population[i] + bias))**2
    return total_error / area


def update_weights(population, profit, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    area = len(population)

    for i in range(area):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*population[i] * (profit[i] - (weight*population[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(profit[i] - (weight*population[i] + bias))

    # We subtract because the derivatives point in direction of steepest ascent
    weight -= (weight_deriv / area) * learning_rate
    bias -= (bias_deriv / area) * learning_rate

    return weight, bias


def train(population, profit, weight, bias, learning_rate, iters):
    cost_history = []
    
    for i in range(iters):
        weight,bias = update_weights(population, profit, weight, bias, learning_rate)
        
        #Calculate cost for auditing purposes
        cost = cost_function(population, profit, weight, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print ("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))

    return weight, bias, cost_history

# Using numpy library
import numpy as np

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost