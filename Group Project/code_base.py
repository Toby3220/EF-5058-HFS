
import datetime as dt
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def read_multi_csv(filenames : list, dt_format : str = '%Y-%m', file_dir = None, name:list = None) -> dict:
    '''
    open multiple csv at the same time and set to a readable format

    Parametersdict
    -----
    - filenames: a list of str of the csv files to be opened 
    - dt_format (optional): format of the date time index
    - file_dir (optional): sub directory, default is current directory
    - name (optional): key values for the dictionary (must match lenght of filenames)
   
    Return
    -----
    df_dict: dictionary of dataframes, with key = name of the file/ provided name, value = data from csv 

    Note
    -----
    input file **MUST** be of specific format:
    - Indexes in the form of dates (corrisponding to Dformat; Default: "%Y-%m") with index column Label "Dates"
    - Column Titles in the form of <Tickers> 
    '''

    #example:(Dformat = %Y-%m" (Default))
    #-  Dates   0111145D    APPL
    #   2000-01 value       value
    #   2001-11 value       value    

    df_dict = {}
    if name is None:
        name = [None]*len(filenames)

    
    for filename,df_name in zip(filenames,name):
        if file_dir:
            tempdf = pd.read_csv(r"{}/{}.csv".format(file_dir,filename))
        else:
            tempdf = pd.read_csv(r"{}.csv".format(filename))

        tempdf["Dates"]=pd.to_datetime(tempdf["Dates"],format=dt_format)
        tempdf.set_index("Dates",inplace=True)

        if df_name:
            df_dict[df_name] = tempdf
        else:
            df_dict[filename] = tempdf
    return df_dict

def past_nvalid_values(df:pd.DataFrame,n:int,df_bool:pd.DataFrame=None)-> dict:
    if df_bool is None:
        df_bool=(np.isnan(df.copy())!=True)
        

    df_dict = {"df":df}
    out_dict = past_nvalid_values_dicts(df_dict,df_bool,n)
    return out_dict["df"]



def past_nvalid_values_dicts(df_dict,df_bool:pd.DataFrame,n:int)-> dict:
    
    # whilst n > 0, we are going to find the previous valid value
    if n>0:
        for df in df_dict.keys():
            tdf =pd.DataFrame(df_dict[df])

            #block out all invalid values, and set df to the previous valid value 
            tdf[df_bool==False]=np.NaN 
            tdf = tdf.fillna(method="ffill").shift(1)
            
            df_dict[df] = tdf
        n-=1
        past_nvalid_values_dicts(df_dict,df_bool,n)
    return df_dict

def dict_transfer(tf_dict,names):
    out_dict={}
    for name in names:
        out_dict[name] = tf_dict[name]
    return out_dict

def information_ratio(data,bench):
    error = data-bench
    ratio = error.mean(skipna=True)/(error.std(skipna=True))
    return ratio*np.sqrt(12)

def run_ml(mldf):
    mldf.dropna(0,inplace=True)
    Y = mldf.pop("RET-ML")
    X = mldf
    X["Intercept"] = 1
    model=sm.OLS(Y,X)

    res = model.fit()
    return model, res
    


class data_obj:
    def __init__(self, tmbool:pd.DataFrame, data_dict:dict):

        self.tmbool = tmbool
        self.data_dict = data_dict

        # self.regulate()

    # def compile_single_stock_df(self, ticker:str, dropna = True, fields = None):
    #     if fields is None:
    #         fields = self.data_dict.keys()

    #     out_df = pd.DataFrame(index=self.tmbool.index, columns=fields)
    #     for field in fields:
    #         tdf = self.data_dict[field]
    #         out_df[field] = tdf[ticker]

    #     if dropna:
    #         out_df.dropna(inplace=True)
    #     return out_df

    ## Key Functions ##

    # calculate z score 
    def calc_z_scores(self, inplace = False, fields=None, use_sample_mean = True, clip="True",clipval=4):
        out_df = {}
        if fields is None:
            fields = self.data_dict.keys()

        for field in fields:
            tdf = self.data_dict[field]
            if use_sample_mean:
                tdf = tdf.subtract(tdf.mean(axis = 1,skipna=True),"index").div(tdf.std(axis = 1,skipna=True),"index")
            else:
                tdf = tdf.div(tdf.std(axis = 1,skipna=True),"index")

            if clip == "True":
                tdf = pd.DataFrame(tdf).clip(-clipval,clipval)
            if clip == "Discard":
                tdf[tdf>=clipval] = 0
                tdf[tdf<=-clipval] = 0
            
            if inplace:
                self.data_dict[field] = tdf 
            else:
                out_df[field] = tdf.copy()

        

        if not inplace:
            new_data = data_obj(self.tmbool.copy(),data_dict=out_df)
            return new_data
    
    def to_port(self,score,tmbool:pd.DataFrame=None):
        if tmbool is None:
            tmbool = self.tmbool

        out = self.regulate(tmbool=tmbool,inplace=False)
        
        port = port_obj(tmbool.copy(),out.data_dict["ret"].copy(),out.data_dict[score].copy())
        return port

    
    ## helper functions ## 
    def calc_ret(self,field,ret_name):
        self.data_dict[ret_name] = (self.data_dict[field]- self.data_dict[field].shift(1))/self.data_dict[field].shift(1)
    
    def get(self,field):
        return self.data_dict[field]
    
    def append(self,field,df):
        self.data_dict[field] = df

    def regulate(self, fields : list = None, tmbool:pd.DataFrame = None, inplace=True):
        if tmbool is None:
            tmbool = self.tmbool

        if fields is None:
            fields = self.tmbool.keys()
        
        out_dict={}
        for key in self.data_dict.keys():
            tdf = self.data_dict[key]

            common_rows = tdf.index.intersection(tmbool.index)
            common_cols = tdf.columns.intersection(tmbool.columns)

            tdf = tdf.loc[common_rows, common_cols]
            tdf[tmbool==False]=np.nan

            if inplace:
                self.data_dict[key] = tdf
            else:
                out_dict[key] = tdf
        
        if not inplace: 
            new_data =  data_obj(tmbool, out_dict)
            return new_data

    def gen_mldf(self,fields:list,tmbool:pd.DataFrame=None):

        data = self.regulate(tmbool=tmbool,inplace=False)
        if tmbool is None:
            tmbool = self.tmbool
        
        mldf = pd.DataFrame(columns=fields)
        tdict = {}
        for field in fields:
            tdf = data.get(field).copy()
            tdf[tmbool!=True]=np.nan
            tdf = tdf.stack(dropna=False)
            mldf[field] = tdf
        return mldf

    
        # mldf = pd.DataFrame(columns=df_dict.keys())
        # out_dict = {}

        # for dfs in df_dict.keys():
        #     cdf = pd.DataFrame(Formatted_dfdict[dfs]).stack(dropna=False)      #cdf = current dataframe
        #     tempdict[dfs] = cdf
 
        # for dfs in Formatted_dfdict.keys():
        #     cdf =tempdict[dfs]
        #     mldf[dfs]=cdf

        # return mldf


    

    
    
    
class port_obj:
    def __init__(self, tmbool:pd.DataFrame, ret:pd.DataFrame, scores:pd.DataFrame):
        self.tmbool = tmbool
        self.ret = ret
        self.scores = scores
    
    def gen_weights_from_score(self,percentile, scores: pd.DataFrame= None, stype:str = "default"):
        if scores is None:
            scores = self.scores.copy()
        else:
            scores = scores.copy()

        pscore = self.calc_percentile(scores)

        lw = pd.DataFrame(0,index = self.tmbool.index, columns= self.tmbool.columns)
        sw = lw.copy()

        if stype == "default":
            lw[pscore>=1-percentile]=pscore[pscore>=1-percentile]-0.5
            lw = lw.divide(lw.sum(1),"index")*1
            lw.fillna(0,inplace=True)

            sw[pscore<=percentile]=pscore[pscore<=percentile]-0.5
            sw = sw.divide(-sw.sum(1),"index")*1
            sw.fillna(0,inplace=True)

            lsw = lw + sw
            self.lw = lw
            self.sw = sw 

        if stype == "equal_weight":
            lw[pscore>=1-percentile]=1
            lw = lw.divide(lw.sum(1),"index")*1
            lw.fillna(0,inplace=True)

            sw[pscore<=percentile]=-1
            sw = sw.divide(-sw.sum(1),"index")*1
            sw.fillna(0,inplace=True)

            lsw = lw + sw

            self.lw = lw
            self.sw = sw 

        elif stype == "bneutral":
            lw[pscore>=1-percentile] = scores[pscore>=1-percentile]
            sw[pscore<=percentile] = scores[pscore<=percentile]

            tdf_l = lw.divide(lw.sum(1).abs(),"index")
            tdf_s = sw.divide(sw.sum(1).abs(),"index")

            lw[lw.sum(1).abs()>=1] = tdf_l
            sw[sw.sum(1).abs()>=1] = tdf_s

            lsw = lw + sw
            self.lw = lw
            self.sw = sw 

        elif stype == "bswing":
            lsw = lw.copy()
            lsw[pscore>=1-percentile] = scores[pscore>=1-percentile]
            lsw[pscore<=percentile] = scores[pscore<=percentile]

            
            tdf = lsw.divide(lsw.sum(1).abs(),"index")
            lsw[lsw.sum(1).abs()>=1] = tdf

            lsw = lsw.clip(lower=-0.1, upper=0.1)

            tdf = lsw.divide(lsw.sum(1).abs(),"index")
            lsw[lsw.sum(1).abs()>=1] = tdf
            # # 4) enforce net exposure limit to [-1, +1]
            # net = lsw.sum(axis=1)
            # # compute scaling factor: 1 if |net|<=1 else 1/|net|
            # scale = net.abs().where(net.abs() <= 1.0, 1.0 / net.abs())
            # # apply scaling row-wise
            # lsw = lsw.mul(scale, axis=0)

            

        
        self.lsw=lsw

        return lsw
    



    def get_port_ret(self,weight:pd.DataFrame=None,bps = 0):

        if weight is None:
            weight = self.lsw

        transaction_fee = (weight.diff().abs().sum(1))*bps/10000
        port_ret = (self.ret*weight.shift(1)).sum(1) - transaction_fee
        cum_ret = (port_ret+1).cumprod()
        return port_ret, cum_ret

    # convert a score value to percentile values
    def calc_percentile(self,scores=None):
        if scores is None:
            scores = self.scores.copy()
        else:
            scores = scores.copy()

       
        for row in scores.index:
            pct = scores.loc[row,:].dropna().rank(pct= True)
            scores.loc[row,pct.index] = pct
        return scores
    
    def calc_evals(self,ret, bench_ret = 0, res: pd.DataFrame = None, Feild_Name = "Current Approach", rf = 0.0025):
        if res is None:
            res = pd.DataFrame(index = ["Annualised Information Ratio","Annualised Sharpe Ratio (rf = 3%)", "Max Drawdown", "Annualised Volatility","Annualised Average Return"])

        res.loc["Annualised Information Ratio", Feild_Name ] = "%.3f" %round(information_ratio(ret,bench_ret),3)
        res.loc["Annualised Sharpe Ratio (rf = 3%)",Feild_Name] = "%.3f" %round(information_ratio(ret,rf),3)
        cret = (ret+1).cumprod(skipna=True)        
        mdd = max(-(cret/cret.cummax()-1))
            
        res.loc["Max Drawdown",Feild_Name] = "%.3f" %round(mdd,3)
        res.loc["Annualised Volatility",Feild_Name] = "%.3f" %round(ret.std(skipna=True)*np.sqrt(12),3)
        res.loc["Annualised Average Return",Feild_Name] = "%.3f"%round(ret.mean(skipna=True)*(12),3)

        return res

            
    
    def quick_plt_diff(self, bench_port, bps = 10, type = "lsw", return_res = False):
        
        if type == "lsw":
            ret, cret = self.get_port_ret(bps=bps)
            bench_ret, bench_cret = bench_port.get_port_ret(bps=bps)
        elif type == "lw":
            ret, cret = self.get_port_ret(self.lw, bps=bps)
            bench_ret, bench_cret = bench_port.get_port_ret(bench_port.lw, bps=bps)
        elif type == "sw":
            ret, cret = self.get_port_ret(self.sw, bps=bps)
            bench_ret, bench_cret = bench_port.get_port_ret(bench_port.sw, bps=bps)
        elif type == "from_ret":
            ret = bench_port
            cret = (bench_port+1).cumprod()

        plt.figure(0)

        plt.plot(cret)
        plt.plot(bench_cret)

        plt.legend(["Current Approach","Benchmark"])
        plt.xlabel("Year")
        plt.ylabel("cumaltive returns (multiples of starting value)")
        plt.title("Factor Portfolio Comparison")
        plt.show()

        res = self.calc_evals(ret,bench_ret)
        res = self.calc_evals(bench_ret,bench_ret,res,"Benchmark")
        # res.loc["Annualised Information Ratio", : ] = [round(information_ratio(ret,bench_ret),3), 0]
        # res.loc["Annualised Sharpe Ratio (rf = 3%)",:] = [round(information_ratio(ret,0.0025),3),round(information_ratio(bench_ret,0.0025),3)]
        
        # mdd = max(-(cret/cret.cummax()-1))
        # bench_mdd = max(-(bench_cret/bench_cret.cummax()-1))
            
        # res.loc["Max Drawdown",:] = [round(mdd,3), round(bench_mdd,3)]
        # res.loc["Annualised Volatility",:] = [round(ret.std(skipna=True)*np.sqrt(12),3), round(bench_ret.std(skipna=True)*np.sqrt(12),3)]
        # res.loc["Annualised Average Return",:] = [round(ret.mean(skipna=True)*(12),3), round(bench_ret.mean(skipna=True)*(12),3)]

        print(res)

        plt.figure(1)
        plt.plot(cret-bench_cret)
        plt.legend(["% points difference (0.1 = 10%)"])
        plt.xlabel("Year")
        plt.ylabel("cumaltive returns (multiples of starting value)")
        plt.title("Approach Return Relative to Momentum Value Benchmark")

        if return_res:
            return res

    def quick_plt_ls(self, bench_port):
        _, tcret = self.get_port_ret(weight=self.lw,bps=10)
        _, bench_tcret = self.get_port_ret(weight=bench_port.lw,bps=10)
        plt.plot(tcret-bench_tcret)

        _, tcret= self.get_port_ret( self.sw, bps=10)
        _, bench_tcret= self.get_port_ret( bench_port.sw, bps=10)

        plt.plot(tcret-bench_tcret)

        plt.legend(["Long Component", "Short Compoenent"])
        plt.xlabel("Year")
        plt.ylabel("% points difference (0.1 = 10%)")
        plt.title("Difference vs Benchmark (Long & Short Component Portfolio)")
        plt.show()
        

# df_dict = read_multi_csv(filenames=["df_annacc_spx","df_memboolG_spx"],name=["annacc","tmbool"])

# past_nvalid_values(df_dict["annacc"],1)

# full_data = data_obj(df_dict.pop("tmbool"),df_dict)
# out_df = full_data.compile("AAPL UW Equity")


df_dict = read_multi_csv(
    filenames=["df_memboolG_spx","df_mcap_spx","df_monthly_spx","df_annsup_spx","df_annest_spx","df_anndat_spx","df_annacc_spx","df_P2BR_spx","df_Beta_spx"],
    name = ["tmbool", "mcap", "price", "annsup", "annest", "anndat", "annacc", "ptbr", "beta"]
)


# shifting regieme by 1 to prevent hindsight bias  
df_dict["annacc"]=df_dict["annacc"].shift(1)
df_dict["annsup"]=df_dict["annsup"].shift(1)

# creating value factor
df_dict["VAL"] = 1/df_dict["ptbr"]

# creating a earnings suprise factor
df_dict["SUP"] =  df_dict["anndat"].shift(1)*df_dict["annsup"]

# creating stock returns 

df_dict["ret"] = (df_dict["price"]/df_dict["price"].shift(1))-1


tdf = pd.DataFrame(df_dict["ret"])
tdf +=1 
tdf = tdf.rolling(10,1).apply(np.prod, raw=True)

df_dict["MOM"] = tdf.shift(2)
df_dict["MOM"] -=1

out_dict = dict_transfer(df_dict,["tmbool","ret","VAL","MOM","SUP"])

# data object 
data = data_obj(out_dict.pop("tmbool").copy(),out_dict)

# convert to z scores
data.calc_z_scores(inplace=True,fields=["VAL","MOM"])

# 
MOM_VAL = (data.get("VAL").fillna(0)+data.get("MOM").fillna(0))/2
data.append("MOM_VAL", MOM_VAL)
data.regulate(["MOM_VAL"])
data.calc_z_scores(inplace=True,fields=["MOM_VAL"])

port = data.to_port("MOM_VAL")

_ = port.gen_weights_from_score(0.1)
momval_ret, momval_cumret = port.get_port_ret()
print("end")