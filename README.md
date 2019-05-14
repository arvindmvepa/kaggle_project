# LANL Earthquake Prediction Kaggle Project

## Setting up EC2

While using EC2 is not necessary to use the project code, in general it's more feasible to run parameter optimization experiments continuously on the cloud rather than on a personal laptop.

### Launching an EC2 Instance
- AMI type: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type (64-bit x86)
- Instance Type: t2.small (t2.micro has memory issues with certain algorithms, instances with more memory are available)
- Storage: 30 GB should be sufficient


Configure Security Group:
- Type: SSH
- Protocal: TCP
- Port Range: 22
- Source: Anywhere
NOTE: When using this setting, may have to remove the other security group rule
Create Key (only has to be done once; later on you can just reference the key you already made):
1. Create a new key name
    1. For example, Arvind named his key `kaggle_project`
2. Download the key pair and keep in a safe location
    1. Cannot access instance without key pair
3. Also, make sure to change permissions of `*pem`
    1. `chmod 400  directory_loc_for_key/*.pem`

### Connect to EC2 instance
**SSH (for MacOS or Linux OS):**
1. Type this command in the terminal: `ssh -i directory_loc_for_key/*.pem -L 8000:localhost:8888 ubuntu@public-dns-name`
    1. Fill in `directory_loc_for_key` with relevant directory and `public-dns-name` with aws public dns name

**Putty (for Windows):**
1. Download Putty (https://putty.org/)
2. Follow directions in https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html (also includes directions for using WinSCP)
3. (Optional) Before opening the putty connection, in order to connect to a remote jupyter process, do this to your putty configuration:
    1. In the Putty side bar, click on “SSH”, and then “Tunnels”
    2. Indicate a reasonable “Source port” (8000) and “Destination” (localhost:8888) and then click on “Add”
    3. Make sure to save your session in Putty so you don’t have re-do all the steps when using Putty.
    4. Click Open on the bottom right of the screen

### Set up EC2 environment
1. (If byobu is not installed, install it) type `byobu` in the terminal
    1. Allows the current session to continue even if you disconnect
2. `sudo apt-get update`
3. `sudo apt-get install -y python3-pip jupyter-core jupyter-notebook e3`
4. (from home directory)
    1. `emacs .bashrc`
    2. add `export PYTHONPATH="$PYTHONPATH:/home/ubuntu/kaggle_project"` to the end of the file
    3. save the file
    4. `source .bashrc`
4. clone the fork of `kaggle_project`:
    1. for example, `git clone https://github.com/USERNAME/kaggle_project.git`

### Run Jupyter Remotely and Access Locally
1. (remotely) `jupyter notebook --no-browser --port=8888`
    1. keep track of the token (you can also add options to not use the token)
2. (locally, via browster) `localhost:8000`
    1. enter the token
3. From here you can run any experiments

## Run Parameter Search Experiments
From what's been tested, in general, running 1 experiment on a t2.small instance works fine. There are memory issues if you try to run multiple processes on the same instance. However, by using multiple instances you can still run the experiment in parallel. Running multiple processes on an instance with more memory hasn't been tested.
1.	Create a yml file with the algorithms and hyper-parameters you would like to search.
    1. Look at test.yml as an example
        1. Outermost indent are the algorithms (keys from `exp.mappings.alg_map`), second indent are the hyper-parameters for the algorithm, and dashed values are the hyper-parameter choices
    1. In order to improve the speed of the search, consider breaking up the yml file into several files
        1. Search different algorithms in each yaml file
2.	Update `main.py` with your specifications:
    1. For example, if the yaml file is “test.yml”, num_searches is 10, and number of folds is 5, line 4 should be `run_experiment_script(params="test.yml", save_results="test_exp.csv", num_searches=10, n_fold=5)`
    2. If you have several yml files, you have to manually edit the main file each time you run main.py
3.	Run `main.py`
    1. You can do this simply with `python3 main.py` on the command line
        1. However, this hogs the command line terminal and you have to create another window on byobu if you want to create multiple processes
    2. A nicer approach if you want to capture the output: `nohup python3 main.py &> log1.txt &`
        1. Make sure to have multiple log files if you are running `main.py` multiple times
    3. A nicer approach if you don’t want to capture the output: `nohup python3 main.py 2>&1 >/dev/null &`



