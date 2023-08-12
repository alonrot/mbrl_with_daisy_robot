import numpy as np
import numpy.random as npr
from datetime import datetime
import pdb

def test1():

    dim = 18
    planning_horizon = 5
    Ntrajectories = 200
    mean_init = npr.normal(size=(dim,))
    mean_zero = np.zeros(dim)
    cov = np.eye(dim)
    trajectory = np.zeros((Ntrajectories,planning_horizon,dim))
    trajectory[0,:] = npr.multivariate_normal(mean=mean_init,cov=cov)
    sample_prev = npr.multivariate_normal(mean=mean_zero,cov=cov,size=(Ntrajectories,))

    Nbenchmarks = 10
    t_bm = np.zeros(Nbenchmarks)
    rejections_vec = np.zeros(Nbenchmarks)
    for bb in range(Nbenchmarks):

        t_init = datetime.utcnow().timestamp()
        cc = -1
        for k in range(1,planning_horizon-1):

            is_forward = False
            while not is_forward:

                # sample = npr.multivariate_normal(mean=mean_zero,cov=cov)
                sample = npr.normal(size=(Ntrajectories,dim))

                is_forward = np.all((sample*sample_prev).sum(axis=1) > 0.)

                cc += 1

            trajectory[:,k+1,:] = trajectory[:,k,:] + sample
            sample_prev = sample
        rejections_vec[bb] = cc

        t_bm[bb] = datetime.utcnow().timestamp() - t_init
        print("t_loop: {0:2.2f} [ms]".format(t_bm[bb]*1000))

    print("rejections_vec:",rejections_vec)
    print("Average time: {0:2.2f} [ms]".format(t_bm.mean()*1000))


def test2():

    dim = 18
    planning_horizon = 5
    Ntrajectories = 200
    mean_init = npr.normal(size=(dim,))
    mean_zero = np.zeros(dim)
    cov = np.eye(dim)
    trajectory = np.zeros((Ntrajectories,planning_horizon,dim))
    trajectory[0,:,:] = npr.multivariate_normal(mean=mean_init,cov=cov,size=(planning_horizon))
    sample_prev = npr.multivariate_normal(mean=mean_zero,cov=cov,size=(dim,))

    Nbenchmarks = 10
    t_bm = np.zeros(Nbenchmarks)
    rejections_vec = np.zeros(Nbenchmarks)
    for bb in range(Nbenchmarks):

        t_init = datetime.utcnow().timestamp()
        cc = -1

        for nt in range(Ntrajectories):

            for k in range(1,planning_horizon-1):

                is_forward = False
                while not is_forward:

                    # sample = npr.multivariate_normal(mean=mean_zero,cov=cov)
                    sample = npr.normal(size=(dim,))

                    is_forward = np.all((sample*sample_prev).sum() > 0.)

                    cc += 1

                trajectory[nt,k+1,:] = trajectory[nt,k,:] + sample
                sample_prev = sample

        rejections_vec[bb] = cc

        t_bm[bb] = datetime.utcnow().timestamp() - t_init
        print("t_loop: {0:2.2f} [ms]".format(t_bm[bb]*1000))

    print("rejections_vec:",rejections_vec)
    print("Average time: {0:2.2f} [ms]".format(t_bm.mean()*1000))


def boundary_check(trajectory_curr,sample):

    dim = trajectory_curr.shape[1]

    bound_low = -0.5*np.ones(dim)
    bound_high = +0.5*np.ones(dim)

    # pdb.set_trace()

    is_out = ((trajectory_curr + sample > bound_high) | (trajectory_curr + sample < bound_low)).sum(axis=1) > 0 # Count how many dimensions are out of boundaries

    return is_out


def test3():

    dim = 18
    planning_horizon = 5
    Ntrajectories = 200
    mean_init = npr.normal(size=(dim,))
    mean_zero = np.zeros(dim)
    cov = np.eye(dim)
    trajectory = np.zeros((Ntrajectories,planning_horizon,dim))
    trajectory[:,0,:] = npr.multivariate_normal(mean=mean_init,cov=cov)
    # pdb.set_trace()
    sample_prev = npr.multivariate_normal(mean=mean_zero,cov=cov,size=(Ntrajectories,))

    Nbenchmarks = 10
    t_bm = np.zeros(Nbenchmarks)
    rejections_vec = np.zeros(Nbenchmarks)
    for bb in range(Nbenchmarks):

        t_init = datetime.utcnow().timestamp()
        cc = -1
        for k in range(0,planning_horizon-1):

            ind_rejected = np.ones(Ntrajectories,dtype=bool)
            sample = np.zeros((Ntrajectories,dim))
            # sample_prev = np.zeros((Ntrajectories,dim)) # not do fi=or fairness with the other implementations
            while ind_rejected.sum() > 0:

                # sample = npr.multivariate_normal(mean=mean_zero,cov=cov)
                sample_subset = npr.normal(size=(ind_rejected.sum(),dim))
                sample[ind_rejected,:] = sample_subset

                # pdb.set_trace()
                # ind_rejected[ind_rejected] = ~((sample[ind_rejected,:]*sample_prev[ind_rejected,:]).sum(axis=1) >= 0.) | boundary_check(trajectory[ind_rejected,k,:],sample[ind_rejected,:]) # Too slow
                ind_rejected[ind_rejected] = ~((sample[ind_rejected,:]*sample_prev[ind_rejected,:]).sum(axis=1) >= 0.)

                cc += 1

            trajectory[:,k+1,:] = trajectory[:,k,:] + sample

            assert np.all((sample*sample_prev).sum(axis=1) >= 0.)
            # assert np.all(~boundary_check(trajectory[ind_rejected,k,:],sample[ind_rejected,:]))

            sample_prev = sample
        rejections_vec[bb] = cc

        t_bm[bb] = datetime.utcnow().timestamp() - t_init
        print("t_loop: {0:2.2f} [ms]".format(t_bm[bb]*1000))

    print("rejections_vec:",rejections_vec)
    print("Average time: {0:2.2f} [ms]".format(t_bm.mean()*1000))



if __name__ == "__main__":
    # test1()

    test2()

    test3()











