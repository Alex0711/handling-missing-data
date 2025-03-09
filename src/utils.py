import pyprojroot

def make_dir_function(dir_name):
  def dir_function(*args):
    return pyprojroot.here().joinpath(dir_name, *args)
  return dir_function