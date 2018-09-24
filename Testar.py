import tensorflow as tf
from Input import pegarBatch
from Config import config
from RedeNeural import RedeNeural


def avaliar():
    iterator = pegarBatch(tamanho_batch=config.batch_size, pasta_dados="./Data/Testar")
    imagens, labels = iterator.get_next()

    cnn = RedeNeural()

    logits = cnn.construir_arquitetura(imagens)

    accuracy = cnn.accuracy(logits, labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        total_batch = 128 // config.batch_size
        avg_acc = 0.
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, './Output/model.ckpt') # /home/samuelehp04/TCC/Output/model.ckpt
        for batch in range(total_batch):
            acc = sess.run(accuracy)
            avg_acc += acc / total_batch
        print("Precisao: {:.5f}".format(avg_acc))
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    avaliar()


if __name__ == "__main__":
    tf.app.run()