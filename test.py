import tensorflow as tf

# #construccion de grafica de computo

# a = tf.constant(3.0)
# b = tf.constant(4.0)
# c= a+b
# print(c)
# sess = tf.Session()
# print(sess.run(c))

# #placeholder siempre requerira del tipo de dato

# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# c = a+b
# print(c)
# sess = tf.Session()
# print(sess.run(c,{a:[1,2],b:[3,4]}))


# #variables en tensores

W = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
modelo_lineal = W * x + b
sess = tf.Session()
#para la utilizacion de variblaes
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run(modelo_lineal,{x:[1,2,3,4]}))


#funcion de costo

#reasignar variables
W_ = tf.assign(W,[-1.0])
b_ = tf.assign(b,[1.0])

sess.run([W_,b_])

y = tf.placeholder(tf.float32)
diferencia_sqr = (modelo_lineal - y)**2

costo = tf.reduce_sum(diferencia_sqr)
print(sess.run(costo,{x:[1,2,3,4], y:[0,-1,-2,-3]}))


#optimizadores
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = opt.minimize(costo)

#resetea las variables al valor inicial
sess.run(init)

##entrenamiento
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))
print(sess.run(costo,{x:[1,2,3,4], y:[0,-1,-2,-3]}))