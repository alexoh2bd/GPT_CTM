import polars as pl
import os
from pathlib import Path
import torch
import tiktoken


class DL:  # DataLoader
    def __init__(self, B, T, dir_path):
        self.B = B
        self.T = T
        self.dir = dir_path
        self.filenames = [f for f in os.listdir(dir_path)]
        self.files_read = 0
        self.enc = tiktoken.get_encoding("gpt2")
        self.total_tokens_skipped = 0
        self.eod_token_id = self.enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        self.curr_pos = 0
        self.data = None
        # print(dir_path)

    # init dataloader with first batch
    def read_next_file(self):
        if self.files_read >= len(self.filenames):
            print(f"Read all {self.files_read}files in Directory.")
            return False
        print(f"Reading file {self.filenames[self.files_read]}")
        all_tokens = []
        self.curr_pos = 0
        curr_df = pl.read_parquet(self.dir / self.filenames[self.files_read])
        # A long list of toke separated by |end of document| tokens
        for i in range(curr_df.shape[0]):
            text = curr_df[i]["text"][0]
            if text and len(text.strip()) > 0:
                tokens = self.enc.encode(text)
                all_tokens.extend(tokens)
                all_tokens.append(self.enc.eot_token)  # add separator at end
        self.data = torch.tensor(all_tokens)
        print("Data Length", len(self.data))
        self.files_read += 1
        return True

    def next_batch(self):
        if self.data is None:
            print("data is none")
            self.read_next_file()
        # if end of the dataframe and skip to next file
        B, T = self.B, self.T
        end_pos = self.curr_pos + (B * T + 1)

        # returns none, none if no data is available
        if end_pos >= len(self.data):
            print(f"end pos {end_pos}> data length{len(self.data)}")
            self.read_next_file()
            return None, None

        print(
            "Next Batch! Current position: ",
            self.curr_pos,
            "end position",
            end_pos,
            "end of row",
            len(self.data),
        )
        buf = self.data[self.curr_pos : end_pos]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.curr_pos += B * T + 1
        return x, y


"""
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / "data" / "raw"
    train_data_dir = raw_data_dir / "train"

    B, T = 2, 512
    num_tokens = 0
    dl = DL(B, T, train_data_dir)
    train_len = len(dl.data)
    x, y = 1, 1
    count = 0
    for _ in range(10):
        dl.read_next_file()
        print(len(dl.data))
        train_len += len(dl.data)
        print(train_len)
    # while x is not None and y is not None:
    #     x, y = dl.next_batch()
    #     count += 1
    # print(count)
    # indices = (dl.data == dl.enc.eot_token).nonzero().flatten()
    # print(dl.data[:10])
    # print("indices of eot", len(indices), indices[:10])
    # 251984184 tokens
"""
