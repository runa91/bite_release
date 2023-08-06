import torch

# code from https://raw.githubusercontent.com/yufu-wang/aves/main/optimization/loss_arap.py


class Arap_Loss():
    '''
    Pytorch implementaion: As-rigid-as-possible loss class
    
    '''

    def __init__(self, meshes, device='cpu', vertex_w=None):

        with torch.no_grad():       # new nadine 

            self.device = device
            self.bn = len(meshes)

            # get lapacian cotangent matrix
            L = self.get_laplacian_cot(meshes)
            self.wij = L.values().clone()
            self.wij[self.wij<0] = 0.

            # get ajacency matrix
            V = meshes.num_verts_per_mesh().sum()
            edges_packed = meshes.edges_packed()
            e0, e1 = edges_packed.unbind(1)
            idx01 = torch.stack([e0, e1], dim=1)
            idx10 = torch.stack([e1, e0], dim=1)
            idx = torch.cat([idx01, idx10], dim=0).t()

            ones = torch.ones(idx.shape[1], dtype=torch.float32).to(device)
            A = torch.sparse.FloatTensor(idx, ones, (V, V))
            self.deg = torch.sparse.sum(A, dim=1).to_dense().long()
            self.idx = self.sort_idx(idx)

            # get edges of default mesh
            self.eij = self.get_edges(meshes)

            # get per vertex regularization strength
            self.vertex_w = vertex_w


    def __call__(self, new_meshes):
        new_meshes._compute_packed()
        
        optimal_R = self.step_1(new_meshes)
        arap_loss = self.step_2(optimal_R, new_meshes)
        return arap_loss


    def step_1(self, new_meshes):
        bn = self.bn
        eij = self.eij.view(bn, -1, 3).cpu()

        with torch.no_grad():
            eij_ = self.get_edges(new_meshes)

            eij_ = eij_.view(bn, -1, 3).cpu()
            wij = self.wij.view(bn, -1).cpu()

            deg_1 = self.deg.view(bn, -1)[0].cpu()  # assuming same topology
            S = torch.zeros([bn, len(deg_1), 3, 3])
            for i in range(len(deg_1)):
                start, end = deg_1[:i].sum(), deg_1[:i+1].sum()
                P  = eij[:, start : end]
                P_ = eij_[:, start : end]
                D = wij[:, start : end]
                D = torch.diag_embed(D)
                S[:, i] = P.transpose(-2,-1) @ D @ P_
                
            S = S.view(-1, 3, 3)

            u, _, v = torch.svd(S)
            R = v @ u.transpose(-2, -1)
            det = torch.det(R)

            u[det<0, :, -1] *= -1
            R = v @ u.transpose(-2, -1)
            R = R.to(self.device)

        return R


    def step_2(self, R, new_meshes):
        R = torch.repeat_interleave(R, self.deg, dim=0)
        Reij = R @ self.eij.unsqueeze(2)
        Reij = Reij.squeeze()
        
        eij_ = self.get_edges(new_meshes)
        arap_loss = self.wij * (eij_ - Reij).norm(dim=1)

        if self.vertex_w is not None:
            vertex_w = torch.repeat_interleave(self.vertex_w, self.deg, dim=0)
            arap_loss = arap_loss * vertex_w

        arap_loss = arap_loss.sum() / self.bn

        return arap_loss


    def get_edges(self, meshes):
        verts_packed = meshes.verts_packed()
        vi = torch.repeat_interleave(verts_packed, self.deg, dim=0)
        vj = verts_packed[self.idx[1]]
        eij = vi - vj
        return eij


    def sort_idx(self, idx):
        _, order = (idx[0] + idx[1]*1e-6).sort()

        return idx[:, order]


    def get_laplacian_cot(self, meshes):
        '''
        Routine modified from :
        pytorch3d/loss/mesh_laplacian_smoothing.py
        '''
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        V, F = verts_packed.shape[0], faces_packed.shape[0]

        face_verts = verts_packed[faces_packed]
        v0, v1, v2 = face_verts[:,0], face_verts[:,1], face_verts[:,2]

        A = (v1-v2).norm(dim=1)
        B = (v0-v2).norm(dim=1)
        C = (v0-v1).norm(dim=1)

        s = 0.5 * (A+B+C)
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / area
        cotb = (A2 + C2 - B2) / area
        cotc = (A2 + B2 - C2) / area
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        ii = faces_packed[:, [1,2,0]]  
        jj = faces_packed[:, [2,0,1]]
        idx = torch.stack([ii, jj], dim=0).view(2, F*3)
        L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))
        L += L.t()
        L = L.coalesce()
        L /= 2.0  # normalized according to arap paper

        return L



