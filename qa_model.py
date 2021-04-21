import numpy as np
import datetime, os


class QAModel:
    def __init__(self, qa_path, rubert):
        self.rubert = rubert
        with open(qa_path) as f:
            lines = f.readlines()

        self.questions = []
        self.answers = []
        for q_a in lines:
            q, a = q_a.split('?')
            self.questions.append(q + '?')
            self.answers.append(a.strip())
        self.idx = 0
        self.session_active = False
        self.user_answers = [''] * len(self.answers)
        self.distances = np.zeros(len(self.questions))
        self.score = 0

    def calculate_score(self):
        self.score = 0
        self.distances = np.zeros(len(self.questions))
        for i in range(len(self.answers)):
            self.distances[i] = self.rubert.calculate_distance(
                [self.answers[i], self.user_answers[i]])

        # TODO Улучшить алгоритм вычисления оценки
        self.score = 1 / self.distances.mean() * 100 - 6
        return self.score

    def save_results(self, name):
        f_path = os.path.join('results', str(name) + '.txt')
        with open(f_path, 'w') as f:
            f.write('%s \n' % datetime.datetime.now())
            for i in range(len(self.user_answers)):
                f.write(self.questions[i])
                f.write(
                    f'{self.user_answers[i]} - {str(self.distances[i])}\n')
            f.write(f'Total: {str(self.score)}')


# Run for self testing
if __name__ == '__main__':
    QA_PATH = 'questions_answers/qa.txt'
    qa = QAModel(QA_PATH, None)
    for i in range(len(qa.answers)):
        print(str(i) + ' ' + qa.questions[i])
        print(qa.answers[i], '\n')
