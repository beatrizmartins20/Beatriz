# create first network with Keras
                from keras.models import Sequential
                from keras.layers import Dense
                import numpy


                # fix random seed for reproducibility
                seed = 7
                numpy.random.seed(seed)


                # load pima indians dataset
                dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")


                # split into input (X) and output (Y) variables
                X = dataset[:,0:8]
                Y = dataset[:,8]


                # create model
                model = Sequential()
                model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
                model.add(Dense(8, init='uniform', activation='relu'))
                model.add(Dense(1, init='uniform', activation='sigmoid'))


                # compile model
                model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])


                # input the dataset into created model
                model.fit(X, Y, nb_epoch=150, batch_size=10)


                # evaluate the model
                scores = model.evaluate(X, Y)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# input image dimensions
              img_rows, img_cols = 28, 28
              # number of convolutional filters to use
              nb_filters = 32
              # convolution kernel size
              kernel_size = (3, 3)
              # size of pooling area for max pooling
              pool_size = (2, 2)

 # define model
              model = Sequential()

              model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                      border_mode='valid',
                                      input_shape=input_shape))
              model.add(Activation('relu'))
              model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
              model.add(Activation('relu'))
              model.add(MaxPooling2D(pool_size=pool_size))
              model.add(Dropout(0.25))


              model.add(Flatten())
              model.add(Dense(128))
              model.add(Activation('relu'))
              model.add(Dropout(0.5))
              model.add(Dense(nb_classes))
              model.add(Activation('softmax'))


              # compile model
              model.compile(loss='categorical_crossentropy',
                            optimizer='adadelta',
                            metrics=['accuracy'])

# input the dataset
              model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=(X_test, Y_test))

              # evaluate model
              score = model.evaluate(X_test, Y_test, verbose=0)

              print('Test score:', score[0])
              print('Test accuracy:', score[1])
            
