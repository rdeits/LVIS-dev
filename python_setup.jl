using PyCall
@pyimport pip
pip.main(["install", "-r", "py-mpc/requirements.txt"])
cd("$(ENV["HOME"])/apps/gurobi701/linux64") do
    run(`$(PyCall.python) setup.py install`)
end