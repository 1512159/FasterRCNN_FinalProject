#Nhap mon TGMT - 15CNTN
#Final Project - Object Detection using Faster RCNN
#Prepare dataset for DETRAC dataset
import xml.etree.ElementTree as ET

def cvtFromXML(inpXML, outTXT):
    tree = ET.parse(inpXML)
    root = tree.getroot()

    of = open(outTXT,"w")

    for frame in root.iter('frame'):
        for target in frame.iter('target'):
            box = target.find('box')
            x1 = float(box.attrib['left'])
            y1 = float(box.attrib['top'])
            x2 = x1 + float(box.attrib['width'])
            y2 = y1 + float(box.attrib['height'])
            tag = target.find('attribute').attrib['vehicle_type']
            of.write("%.2f %.2f %.2f %.2f %s\n" % (x1,y1,x2,y2,tag))
        of.write('###\n')
    of.close()

def main():
    cvtFromXML('mvi_20011.xml','mvi_20011.txt')
if __name__ == '__main__':
    main()
