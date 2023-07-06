'''行为克隆'''

import zip_array
from config import CONFIG


if CONFIG['use_redis']:
    import my_redis, redis
    import zip_array

if CONFIG['use_frame'] == 'paddle':
    from paddle_net import PolicyValueNet
elif CONFIG['use_frame'] == 'pytorch':
    from pytorch_net import PolicyValueNet

move_id2move_action, move_action2move_id = get_all_legal_moves()

class behaviour_learning:
    # 从数据集读入每场比赛的WXF格式走子，程序自动模拟转换成训练集
    # 实际比赛数据 -> zip(state,move,winners)
    # state:9*10*9特征平面，move：长度为2086的pi向量，winners:每局比赛的赢家

    def __init__(self):
        self.iters=0
        self.board=Board()
        self.board.init_board()
        self.chess_dict = dict(R=0, r=0, H=1, h=1, E=2, e=2, A=3, a=3, K=4, k=4, C=5, c=5, P=6, p=6)
        self.move_list = pd.read_csv('moves.csv')
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)

    # 根据输入的move_id和当前局面寻找走子的起点坐标
    def find_locate(self,player,move_id):

        # 红方的情况
        if player == 'red':
            # WXF格式第二位为-，代表同一列靠后的那个棋子
            if move_id[1] == '-':
                #对兵特判
                if self.chess_dict[move_id[0]]==6:
                    for i in range(10):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == 1:
                                for k in range(i+1,10):
                                    if self.board.current_state()[self.chess_dict[move_id[0]]][k][j] == 1:
                                        return i,j
                else:
                    for i in range(10):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == 1:
                                return i,j

            # WXF格式第二位为+，代表同一列靠前的那个棋子
            elif move_id[1] == '+':
                # 对兵特判
                if self.chess_dict[move_id[0]] == 6:
                    for i in range(9,-1,-1):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == 1:
                                for k in range(i-1,-1,-1):
                                    if self.board.current_state()[self.chess_dict[move_id[0]]][k][j] == 1:
                                        return i,j
                else:
                    for i in range(9,-1,-1):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == 1:
                                return i,j

            # WXF格式第二位为数字，直接在相应的列寻找
            else:
                for i in range(10):
                    #print((self.board.current_state()[self.chess_dict[move_id[0]]][i][1]))
                    if self.board.current_state()[self.chess_dict[move_id[0]]][i][int(move_id[1])-1] == 1:
                        return i, int(move_id[1])-1

        # 黑方的情况
        if player == 'black':
            # WXF格式第二位为+，代表同一列靠前的那个棋子
            if move_id[1] == '+':
                if self.chess_dict[move_id[0]] == 6:
                    for i in range(10):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == -1:
                                for k in range(i+1,10):
                                    if self.board.current_state()[self.chess_dict[move_id[0]]][k][j] == -1:
                                        return i, j
                else:
                    for i in range(10):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == -1:
                                return i,j

            # WXF格式第二位为-，代表同一列靠后的那个棋子
            elif move_id[1] == '-':
                if self.chess_dict[move_id[0]] == 6:
                    for i in range(9,-1,-1):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == -1:
                                print(i,j)
                                for k in range(i-1,-1,-1):
                                    if self.board.current_state()[self.chess_dict[move_id[0]]][k][j] == -1:
                                        return i, j
                else:
                    for i in range(9,-1,-1):
                        for j in range(9):
                            if self.board.current_state()[self.chess_dict[move_id[0]]][i][j] == -1:
                                return i,j

            # WXF格式第二位为数字，直接在相应的列寻找
            else:
                for i in range(10):
                    if self.board.current_state()[self.chess_dict[move_id[0]]][i][9-int(move_id[1])] == -1:
                        return i, 9-int(move_id[1])

    # 根据输入的move_id和当前局面寻找走子的终点坐标
    def move(self,startX,startY,player,move_id):

        # 红方情况
        if player == 'red':
            # 马的走子
            if self.chess_dict[move_id[0]] == 1:
                endY=int(move_id[3])-1
                if move_id[2] == '+':
                    endX=startX+(3-abs(startY-endY))
                else:
                    endX=startX-(3-abs(startY-endY))

            # 象的走子
            elif self.chess_dict[move_id[0]] == 2:
                endY=int(move_id[3])-1
                if move_id[2] == '+':
                    endX=startX+2
                else:
                    endX=startX-2

            # 士的走子
            elif self.chess_dict[move_id[0]] == 3:
                endY=int(move_id[3])-1
                if move_id[2] == '+':
                    endX=startX+1
                else:
                    endX=startX-1

            # 其他棋子的走子
            else:
                if move_id[2] == '+':
                    endY=startY
                    endX=startX+int(move_id[3])
                elif move_id[2] == '-':
                    endY = startY
                    endX = startX - int(move_id[3])
                else:
                    endY = int(move_id[3])-1
                    endX = startX

        # 黑方情况
        if player == 'black':
            # 马的走子
            if self.chess_dict[move_id[0]] == 1:
                endY=9-int(move_id[3])
                if move_id[2] == '+':
                    endX=startX-(3-abs(startY-endY))
                else:
                    endX=startX+(3-abs(startY-endY))

            # 象的走子
            elif self.chess_dict[move_id[0]] == 2:
                endY = 9-int(move_id[3])
                if move_id[2] == '+':
                    endX = startX - 2
                else:
                    endX = startX + 2

            # 士的走子
            elif self.chess_dict[move_id[0]] == 3:
                endY = 9-int(move_id[3])
                if move_id[2] == '+':
                    endX = startX - 1
                else:
                    endX = startX + 1

            # 其他棋子的走子
            else:
                if move_id[2] == '+':
                    endY = startY
                    endX = startX - int(move_id[3])
                elif move_id[2] == '-':
                    endY = startY
                    endX = startX + int(move_id[3])
                else:
                    endY = 9-int(move_id[3])
                    endX = startX

        return endX,endY

    def get_equi_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        extend_data = []
        # 棋盘状态shape is [9, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            extend_data.append(za.zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])
            state = state.transpose([1, 2, 0])
            for i in range(10):
                for j in range(9):
                    state_flip[i][j] = state[i][8 - j]
            state_flip = state_flip.transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
            extend_data.append(za.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return extend_data

    # 读入每局数据并模拟，生成数据集传入pkl文件
    def simulate(self):
        if os.path.exists(CONFIG['train_data_buffer_path']):
            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                data_file = pickle.load(data_dict)
                self.iters = data_file['iters']
                del data_file
            print('已载入数据')
        else:
            self.iters += 1

        play_data=[]
        game_info=[]

        #循环开始
        for id in range(57380690+self.iters, 57390690):
            print("读取第%d轮对局信息"%(self.iters))
            self.iters += 1

            #数据预处理，红黑交替存储在move_MXF_play中
            states, move_probs, winners = [], [], []
            move_MXF = self.move_list[self.move_list['gameID'] == id]
            move_MXF_red = list(move_MXF[move_MXF['side'] == 'red']['move'])
            move_MXF_black = list(move_MXF[move_MXF['side'] == 'black']['move'])
            move_MXF_play = []
            self.board.init_board()
            for step in range(len(move_MXF_red)):
                move_MXF_play.append(move_MXF_red[step])
                if step < len(move_MXF_black):
                    move_MXF_play.append(move_MXF_black[step])

            #判断胜负
            if len(move_MXF_red) > len(move_MXF_black):
                winner='red'
            else:
                winner='black'

            #对局开始
            for index in range(len(move_MXF_play)):
                if index%2==0:
                    player='red'
                else:
                    player='black'

                #收集数据
                startX,startY=self.find_locate(player,move_MXF_play[index])
                endX,endY=self.move(startX,startY,player,move_MXF_play[index])
                move=str(startX)+str(startY)+str(endX)+str(endY)
                self.board.do_move(move_action2move_id[move])
                states.append(self.board.current_state())
                move_probs_temp = np.zeros(2086)
                move_probs_temp[move_action2move_id[move]]=1.0
                move_probs.append(move_probs_temp)
                if player==winner:
                    winners.append(1.0)
                else:
                    winners.append(-1.0)

            play_data=zip(states, move_probs, winners)
            play_data = list(play_data)[:]
            play_data=self.get_equi_data(play_data)
            game_info.extend(play_data)
            #print(play_data)
            #print(str(id)+'   '+str(index)+'   '+move_MXF_play[index]+'  '+str(move_action2move_id[move]))

        #存储数据
        if os.path.exists(CONFIG['train_data_buffer_path']):
            while True:
                try:
                    with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                        data_file = pickle.load(data_dict)
                        self.data_buffer = deque(maxlen=self.buffer_size)
                        self.data_buffer.extend(data_file['data_buffer'])
                        del data_file
                        self.data_buffer.extend(game_info)
                    print('成功载入数据')
                    break
                except:
                    print('载入数据失败')
        else:
            self.data_buffer.extend(game_info)

        data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
        with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
            pickle.dump(data_dict, data_file)


a=behaviour_learning()
a.simulate()

