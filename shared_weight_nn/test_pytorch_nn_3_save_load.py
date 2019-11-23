torch.save(Net, "net.pth")

Net = torch.load("net.pth")

print(Net)


for key in Net.state_dict():
  print(key, Net.state_dict()[key])

torch.save(Net.state_dict(), "net_state_dict.pth")

Net.load_state_dict(torch.load("net_state_dict.pth"))