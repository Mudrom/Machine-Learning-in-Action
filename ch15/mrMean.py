from mrjob.job import MRJob
from mrjob.step import MRStep

class MRmean(MRJob):
    def __init__(self, *args, **kwargs):
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0

    def map(self, key, val):  # needs exactly 2 arguments 按照key值进行排序，相同key发送给一个reducer
        if False: yield
        inVal = float(val)
        self.inCount += 1
        self.inSum += inVal
        self.inSqSum += inVal * inVal # 误差累积

    def map_final(self):
        mn = self.inSum / self.inCount
        mnSq = self.inSqSum / self.inCount
        yield (1, [self.inCount, mn, mnSq])

    def reduce(self, key, packedValues):
        cumVal = 0.0;
        cumSumSq = 0.0;
        cumN = 0.0
        for valArr in packedValues:  # get values from streamed inputs
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal / cumN
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
        yield (mean, var)  # emit mean and var

    def steps(self):
        return ([MRStep(mapper=self.map, mapper_final=self.map_final, \
                         reducer=self.reduce, )])


if __name__ == '__main__':
    MRmean.run()
