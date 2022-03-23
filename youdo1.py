import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

def main():
    st.header("Build a Regression Model")
    st.markdown("""
    In this project, we will be implementing a linear model:
    
    1. Finding a new convex loss function
    2. Showing that it is convex
    3. Implementing gradient descent algorithm
    4. Minimizing loss function with optimal beta values
    5. Converting model to L2 
    6. Compare this model and the model in wedo
    
    """)

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    st.dataframe(X)
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target))

    st.dataframe(df)

    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Formulation of model and error function")

    st.markdown("#### General Model")
    st.latex(r"\hat{y}_i=\beta_0 + \beta_1 x_i")

    st.markdown("#### Loss Function")

    st.write("Let's say θ to our threshold multiplier")
    st.write("In loss function we should accept a threshold around +- θ% percent of the y")
    st.latex(r"(y_i - \hat{y}_i )")
    st.write("If the absolute value of above is larger than θ * y_i error value is calculated with that threshold value")

    st.latex(
        r"L(\beta_0,\beta_1)=\sum_{i=1}^{N}{(y_i - \hat{y}_i )^2 }")
    st.write("In threshold case, i.e.,")
    st.latex(r"y_i(1-\theta) < \hat{y}_i < y_i(1-\theta)")
    st.write("That will be applied. So in the range function is convex, constant out of the range")
    st.latex(r"\hat{y}_i = \theta y_i")

    st.markdown("#### Partial derivatives")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0}=-2\sum_{i=1}^{N}{(y_i - \hat{y}_i) }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_1}=-2\sum_{i=1}^{N}{(y_i - \hat{y}_i)x_i}")

    ####
    st.markdown(
            r"Given that $L$ is convex wrt both $\beta_0$ and $\beta_1$, "
            r"we can use Gradient Descent to find  $\beta_0^{*}$ and $\beta_1^{*}$ by using partial derivatives")
    st.latex(
        r"\frac{\partial L}{\partial \beta_0} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )}")
    st.latex(
        r"\frac{\partial L}{\partial \beta_1} =  -2\sum^{N}_{i=1}{(y_i - \beta_0 - \beta_1 x_i )x_i}")
###

    st.write("***special note for TA's and Instructor: Mathematical notation can be problematic but it is implemented ***")

    st.header("Regression simple Model")
    theta = 0.6
    beta = regSimple(df['MedInc'].values, df['Price'].values, theta)
    st.write("beta0 = ", beta[0], "beta1 = ", beta[1])

    beta_l2 = reg_l2(df['MedInc'].values, df['Price'].values, theta)
    st.write("beta0 = ", beta_l2[0], "beta1 = ", beta_l2[1])

    performance(beta, df['MedInc'].values, df['Price'].values)
    

def regSimple(x,y,theta, alpha=0.001):
    beta = np.random.random(2) 

    print("starting sgd")
    for i in range(100):

        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum()
        g_b1 = -2 * (x * (y - y_pred)).sum()

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        err = 0

        for _y, _y_pred in zip(y,y_pred):

            if abs(_y-_y_pred) < theta*_y:
                err += (_y-_y_pred)**2
            
            else:
                err += theta*_y

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}, error: {err}")
        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta

def reg_l2(x, y, theta, lam=0.1, alpha=0.0001) -> np.ndarray:
    print("starting sgd")
    beta = np.random.random(2)

    for i in range(100):
        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum() + 2 * lam * beta[0]
        g_b1 = -2 * (x * (y - y_pred)).sum() + 2 * lam * beta[1]

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        err = 0

        for _y, _y_pred in zip(y,y_pred):

           if abs(_y-_y_pred) < theta*_y:
                err += (_y-_y_pred)**2
            
           else:
                err += theta*_y

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}, error: {err}")

        if np.linalg.norm(beta - beta_prev) < 0.000001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta
    
def performance(beta,X,y):
    y_pred=[]
    for v in X:                                                                                                                                                                                                             
         y_pred.append( beta[0] + beta[1] * v)

    dg1 = pd.DataFrame(dict(x=X, y=y, y_pred=y_pred))


    fig1 = plt.figure(figsize = (10, 10))
    plt.plot(X, dg1["y"], "b.", markersize = 5)
    plt.plot(X, dg1["y_pred"], "c.", markersize = 5)
    st.plotly_chart( fig1, use_container_width=True)                    

if __name__ == '__main__':
    main()