
resolution     = '15000_1';
train_ids      = 1;
val_ids        = [1;2;3];
trainfinal_ids = [1,2];
valfinal_ids   = [3];
test_ids       = 4;

data = struct();
data.resolution     = resolution;
data.train_ids      = train_ids;
data.val_ids        = val_ids;
data.trainfinal_ids = trainfinal_ids;
data.valfinal_ids   = valfinal_ids;
data.test_ids       = test_ids;
data.nrun           = 1;

r = script_sda_logdetector( data )


