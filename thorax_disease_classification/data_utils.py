from PIL import Image
from itertools import chain

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class NIH_Dataset(Dataset):
  """
  Main Dataset class to represent NIH Chest X-Ray data
  ...
  Attributes
  ----------
  data_df: pandas.DataFrame
    A DataFrame that needs to at least have the following two columns: 
      1. "path": Holds chest x-ray image locations
      2. "disease_vec": Holds multilabel one-hot encoded vector
  transform: torchvision.transform.Compose
    Image transformations that will be applied to each image when calling "__getitem__"
  """
  def __init__(
    self, 
    data_df,
    transform=None
    ) -> Dataset:
    """
    Parameters
    ----------
    data_df: pandas.DataFrame
      A DataFrame that needs to at least have the following two columns: 
        1. "path": Holds chest x-ray image locations
        2. "disease_vec": Holds multilabel one-hot encoded vector
    transform: torchvision.transform.Compose
      Image transformations that will be applied to each image when calling "__getitem__"
    """
    self.data_df = data_df
    self.transform = transform 

  def __len__(
    self
    ) -> int:
    return len(self.data_df)

  def __getitem__(
    self, 
    idx
    ) -> Tuple[torch.Tensor, np.ndarray]:
    img_file = self.data_df['path'].iloc[idx]
    img = Image.open(img_file).convert('RGB')
    label = np.array(self.data_df.iloc[:,-1].iloc[idx], dtype=float)
    if self.transform:
        img = self.transform(img)

    return img, label

def load_and_preprocess_data_df(
    data_df_root_path: str,
    num_data_samples: int = 11000
    ) -> pd.DataFrame:
  """
  Loads and preprocesses NIH DataFrame that holds information of the data, e.g. image labels.
  """

  # Load the DataFrame
  all_xray_df = pd.read_csv(
      os.path.join(
          data_df_root_path,
          'Data_Entry_2017.csv'
          )
      )
  
  # Extract all image paths
  all_image_paths = {
      os.path.basename(x): x for x in 
      glob(os.path.join(data_df_root_path, 'images*', '*', '*.png'))
      }

  # Add image file name column
  all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
  
  # Count labels
  all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))  
  all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
  all_labels = [x for x in all_labels if len(x)>0]
  for c_label in all_labels:
      if len(c_label)>1: # leave out empty labels
          all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
  # MIN_CASES = 1000
  # all_labels = [
  #     c_label for c_label in all_labels if all_xray_df[c_label].sum() > MIN_CASES
  #     ]
  sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
  sample_weights /= sample_weights.sum()
  preprocessed_xray_df = all_xray_df.sample(num_data_samples, weights=sample_weights)
  # Create one-hot encoding of the vector
  preprocessed_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
  return preprocessed_xray_df

def inv_data_transform(
    img: torch.Tensor
    ) -> torch.Tensor:
  """
  Helper function that un-normalizes torch image tensor
  """
  img = img.permute(1,2,0)
  img = img * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])
  return img

def get_data_loaders(
    data_df_root_path: str,
    num_data_samples: int = 11000,
    global_train_frac: float = 0.8,
    local_train_frac: float = 0.8,
    batch_size: int = 32,
    num_clients: int = 10
    ) -> Tuple[Tuple[DataLoader, ...], Tuple[DataLoader, ...], DataLoader, List]:

  """
  Main function to create DataLoaders of the NIH X-Ray data set.
  
  Parameters
  ----------
  data_df_root_path: str
    This should be the path to the root directory of the NIH data set that, 
    at least, includes the img-file directories and the panda.DataFrame 
    "Data_Entry_2017.csv". The direcotry should have the following structure:

    NIH
      |
      images_0**
      ...
      Data_Entry_2017.csv
      ...
      train_val_list.txt

  num_data_samples: int
    The total number of data points that we want to sample for our data set.
    Maximum is 112120.

  global_train_frac: float
    The fraction of points used for the global training data set, e.g. the data set that will be later split into local client data sets.

  local_train_frac: float
      The fraction of points on each local client used for training.

  batch_size: int
    Batch size on each client

  num_clients: int
    Total number of clients in the FL setup

  Returns
  -------
  trainloaders: torch.utils.data.DataLoader

  """
  
  preprocessed_xray_df = load_and_preprocess_data_df(
      data_df_root_path,
      num_data_samples
  )

  train_data_transform = T.Compose([
    T.RandomRotation((-20,+20)),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])

  test_data_transform = T.Compose([
    T.Resize((512,512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])
  
  train_df, test_df = train_test_split(
      preprocessed_xray_df,
      test_size = 1 - global_train_frac,
      random_state = 42,
      stratify = preprocessed_xray_df['Finding Labels'].map( lambda x: x[:4])
  )

  trainset = NIH_Dataset(
      train_df,
      transform = train_data_transform
      )
  
  testset = NIH_Dataset(
      test_df,
      transform = test_data_transform
      )

  # Split training set into `num_clients` partitions to simulate different local datasets
  partition_size = len(trainset) // num_clients
  lengths = [partition_size] * num_clients
  lengths[-1] += len(trainset) - np.sum(lengths)
  datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

  # Split each partition into train/val and create DataLoader
  trainloaders = []
  valloaders = []
  for ds in datasets:      
      len_train = math.ceil(len(ds) * local_train_frac)
      len_val = len(ds) - math.ceil(len(ds) * local_train_frac)
      lengths = [len_train, len_val]
      ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
      trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last = True))
      valloaders.append(DataLoader(ds_val, batch_size=batch_size, drop_last = True))
  testloader = DataLoader(testset, batch_size=batch_size)

  return trainloaders, valloaders, testloader, all_labels    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])
  
  train_df, test_df = train_test_split(
      preprocessed_xray_df,
      test_size = 1 - global_train_frac,
      random_state = 42,
      stratify = preprocessed_xray_df['Finding Labels'].map( lambda x: x[:4])
  )

  trainset = NIH_Dataset(
      train_df,
      transform = train_data_transform
      )
  
  testset = NIH_Dataset(
      test_df,
      transform = test_data_transform
      )

  # Split training set into `num_clients` partitions to simulate different local datasets
  partition_size = len(trainset) // num_clients
  lengths = [partition_size] * num_clients
  lengths[-1] += len(trainset) - np.sum(lengths)
  datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

  # Split each partition into train/val and create DataLoader
  trainloaders = []
  valloaders = []
  for ds in datasets:      
      len_train = math.ceil(len(ds) * local_train_frac)
      len_val = len(ds) - math.ceil(len(ds) * local_train_frac)
      lengths = [len_train, len_val]
      ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
      trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last = True))
      valloaders.append(DataLoader(ds_val, batch_size=batch_size, drop_last = True))
  testloader = DataLoader(testset, batch_size=batch_size)

  return trainloaders, valloaders, testloader, all_labels

def inv_data_transform(
    img: torch.Tensor
    ) -> torch.Tensor:
  """
  Helper function that un-normalizes torch image tensor
  """
  img = img.permute(1,2,0)
  img = img * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])
  return img