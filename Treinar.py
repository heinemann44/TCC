import tensorflow as tf
from RedeNeural import RedeNeural
from Input import pegarBatch
from Config import config


def treinar():

    iterator = pegarBatch(tamanho_batch=config.batch_size, pasta_dados="./Data/Treinar")
    imagens, labels = iterator.get_next()
    tf.summary.image("imagens", imagens, config.batch_size)
    cnn = RedeNeural()

    logits = cnn.construir_arquitetura(imagens)
    custo = cnn.custo(logits, labels)
    treinamento = cnn.treinar(custo)
    accuracy = cnn.accuracy(logits, labels)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('Output/treinamento', sess.graph)
        total_batch = 10000//config.batch_size
        print("epoch " + str(config.epoch))
        count = 0
        for i in range(config.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for batch in range(total_batch):
                count += 1
                merge = tf.summary.merge_all()
                sumario, _ = sess.run([merge,treinamento])
                summary_writer.add_summary(sumario, count)
                loss, acc = sess.run([custo, accuracy])
                avg_cost += loss / total_batch
                avg_acc += acc / total_batch
            print("Iter " + str(i) + ", Loss= " + \
                  "{:.6f}".format(avg_cost) + ", Training Accuracy= " + \
                  "{:.5f}".format(avg_acc))
        save_path= saver.save(sess, "Output/model.ckpt")
        print("Model saved in path: %s" % save_path)

def main(argv=None):
    treinar()


if __name__ == "__main__":
    tf.app.run()
