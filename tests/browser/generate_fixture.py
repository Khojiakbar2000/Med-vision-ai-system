"""Synthetic spatial phantom only; output must be a temporary external folder."""
import sys, struct, math
from pathlib import Path
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.sequence import Sequence
root=Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
mode=sys.argv[2] if len(sys.argv)>2 else 'single'
if mode=='malformed':
    (root/'invalid.bin').write_bytes(b'not a DICOM image')
    raise SystemExit(0)
for instance in range(1 if mode in ('multi', 'stack') else 16):
    meta=FileMetaDataset(); meta.TransferSyntaxUID='1.2.840.10008.1.2.1'; meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.4.1' if mode=='multi' else '1.2.840.10008.5.1.4.1.1.4'; meta.MediaStorageSOPInstanceUID=f'2.25.900001{instance}'
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=meta.MediaStorageSOPClassUID; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID='2.25.900002'; ds.SeriesInstanceUID='2.25.900003'; ds.FrameOfReferenceUID='2.25.900004'
    if mode=='missinguid': del ds.SeriesInstanceUID
    ds.PatientName='SYNTHETIC_PRIVACY_SENTINEL'; ds.PatientID='SYNTHETIC_PRIVACY_SENTINEL'; ds.Modality='MR'; ds.Rows=64; ds.Columns=64; ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'; ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0; ds.RescaleSlope=1; ds.RescaleIntercept=0; ds.WindowCenter=500; ds.WindowWidth=1000; ds.InstanceNumber=instance+1
    if mode=='multi':
        ds.NumberOfFrames=16
        shared=Dataset(); orientation=Dataset(); orientation.ImageOrientationPatient=[1,0,0,0,1,0]; shared.PlaneOrientationSequence=Sequence([orientation]); spacing=Dataset(); spacing.PixelSpacing=[1,1]; spacing.SliceThickness=2; shared.PixelMeasuresSequence=Sequence([spacing]); ds.SharedFunctionalGroupsSequence=Sequence([shared]); frames=[]
        for i in range(16):
            frame=Dataset(); position=Dataset(); position.ImagePositionPatient=[0,0,i*2]; frame.PlanePositionSequence=Sequence([position]); frames.append(frame)
        ds.PerFrameFunctionalGroupsSequence=Sequence(frames)
    else:
        ds.ImageOrientationPatient=[1,0,0,0,1,0]; ds.ImagePositionPatient=[0,0,instance*2]; ds.PixelSpacing=[1,1]; ds.SliceThickness=2
    pixels=[]
    for z in range(16 if mode=='multi' else 1):
        for y in range(64):
            for x in range(64):
                d=((x-32)/24)**2+((y-32)/28)**2
                pixels.append(0 if d>1 else 300+(400 if (x-26)**2+(y-32)**2<80 else 0)+(z if mode=='multi' else instance)*10)
    ds.PixelData=struct.pack('<'+'H'*len(pixels),*pixels)
    ds.save_as(root/f'{instance}.dcm',enforce_file_format=True)
