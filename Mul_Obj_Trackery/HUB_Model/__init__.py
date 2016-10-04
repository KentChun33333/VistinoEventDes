import dlib


class StaticModel(object):
    def __init__(self):
        pass

    def detect(sefl):
        pass

    def show_detect(self):
        pass


class Tracker(object):
    def __init__(self):
        self.tracker=dlib.correlation_tracker()
    
    def start_track(self, img, tarBox):
        startX, startY, endX, endY = tarBox
        d = dlib.rectangle(startX, startY, endX, endY)
        return self.tracker.start_track(img, d)
    
    def update(self,*arg, **kwargs):
        return self.tracker.update(*arg, **kwargs)

    def get_position(self,*arg, **kwargs):
        d = self.tracker.get_position(*arg, **kwargs)
        startX, startY, endX, endY = self.to_int(d.left(), d.top(),d.right(),d.bottom())
        return (startX, startY, endX, endY)

    def to_int(self, *args):
        result=[]
        for i in args:
            result.append(int(i))
        return result
